"""Pure FAO-56-based daily irrigation advice for sugarcane."""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CropParameters:
    t_base_c: float = 12.0
    t_upper_c: float = 38.0
    kc_initial: float = 0.40
    kc_midseason: float = 1.25
    kc_endseason: float = 0.75
    gdd_initial: float = 450.0
    gdd_development: float = 1200.0
    gdd_midseason: float = 3200.0
    gdd_endseason: float = 4000.0
    root_depth_min_m: float = 0.30
    root_depth_max_m: float = 1.20
    depletion_fraction: float = 0.65


@dataclass(frozen=True)
class FarmConfig:
    farm_id: str
    latitude_deg: float
    longitude_deg: float
    area_m2: float
    planting_date: dt.date
    flow_lph: float = 4.0
    emitters_per_acre: int = 4000
    elevation_m: float = 600.0
    geometry_geojson: dict | None = None

    @property
    def drip_application_rate_mm_per_hour(self) -> float:
        if self.area_m2 <= 0:
            raise ValueError("Farm area must be greater than zero")
        area_acres = self.area_m2 / 4046.856
        return self.emitters_per_acre * area_acres * self.flow_lph / self.area_m2


@dataclass(frozen=True)
class DailyWeather:
    date: dt.date
    tmax_c: float
    tmin_c: float
    rain_mm: float = 0.0
    actual_vapor_pressure_kpa: float | None = None
    solar_radiation_mj_m2: float | None = None
    wind_speed_2m_m_s: float | None = None


@dataclass(frozen=True)
class CropState:
    cumulative_gdd: float
    depletion_mm: float
    last_processed_date: dt.date | None = None


@dataclass(frozen=True)
class IrrigationAdvice:
    farm_id: str
    date: dt.date
    status: str
    irrigate_now: bool
    recommended_net_mm: float
    recommended_gross_mm: float
    recommended_liters: float
    recommended_drip_hours: float
    depletion_mm: float
    trigger_mm: float
    days_to_trigger: int | None
    et0_mm_day: float
    et0_method: str
    crop_coefficient: float
    crop_coefficient_source: str
    root_depth_m: float
    crop_stress_coefficient: float
    runoff_mm: float
    deep_percolation_mm: float
    next_state: CropState


SUGARCANE = CropParameters()
SOIL_TAW_MM_PER_M = 180.0
SCS_CURVE_NUMBER = 85.0
NDVI_SOIL = 0.20
NDVI_FULL = 0.85
IRRIGATION_EFFICIENCY = 0.90
MAD_TRIGGER = 0.50
LEAD_DAYS = 1.0
MAX_NET_EVENT_MM = 45.0


def saturation_vapor_pressure(t_c: float) -> float:
    """Saturation vapor pressure in kPa (FAO-56 equation 11)."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def growing_degree_days(tmax_c: float, tmin_c: float, crop: CropParameters = SUGARCANE) -> float:
    """Daily GDD with temperatures clipped to the crop's base and upper limits."""
    maximum = min(tmax_c, crop.t_upper_c)
    minimum = max(tmin_c, crop.t_base_c)
    return max((maximum + minimum) / 2.0 - crop.t_base_c, 0.0)


def extraterrestrial_radiation(latitude_deg: float, day_of_year: int) -> float:
    """Daily extraterrestrial radiation in MJ m-2 (FAO-56 equation 21)."""
    latitude = math.radians(latitude_deg)
    inverse_distance = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
    solar_declination = 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)
    argument = max(-1.0, min(1.0, -math.tan(latitude) * math.tan(solar_declination)))
    sunset_angle = math.acos(argument)
    return (24 * 60 / math.pi) * 0.0820 * inverse_distance * (
        sunset_angle * math.sin(latitude) * math.sin(solar_declination)
        + math.cos(latitude) * math.cos(solar_declination) * math.sin(sunset_angle)
    )


def reference_evapotranspiration_penman_monteith(
    tmax_c: float,
    tmin_c: float,
    actual_vapor_pressure_kpa: float,
    solar_radiation_mj_m2: float,
    wind_speed_2m_m_s: float,
    elevation_m: float,
    latitude_deg: float,
    day_of_year: int,
) -> float:
    """Daily FAO-56 Penman-Monteith reference ET0 in mm/day."""
    mean_temperature = (tmax_c + tmin_c) / 2.0
    pressure = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    psychrometric_constant = 0.000665 * pressure
    slope = 4098.0 * saturation_vapor_pressure(mean_temperature) / (mean_temperature + 237.3) ** 2
    saturation_pressure = (saturation_vapor_pressure(tmax_c) + saturation_vapor_pressure(tmin_c)) / 2.0
    actual_vapor_pressure_kpa = min(actual_vapor_pressure_kpa, saturation_pressure)
    radiation_toa = extraterrestrial_radiation(latitude_deg, day_of_year)
    radiation_clear_sky = (0.75 + 2e-5 * elevation_m) * radiation_toa
    solar_radiation_mj_m2 = min(solar_radiation_mj_m2, radiation_clear_sky)
    net_shortwave = (1 - 0.23) * solar_radiation_mj_m2
    relative_radiation = min(solar_radiation_mj_m2 / radiation_clear_sky, 1.0) if radiation_clear_sky > 0 else 0.5
    net_longwave = (
        4.903e-9
        * (((tmax_c + 273.16) ** 4 + (tmin_c + 273.16) ** 4) / 2.0)
        * (0.34 - 0.14 * math.sqrt(max(actual_vapor_pressure_kpa, 0.0)))
        * (1.35 * relative_radiation - 0.35)
    )
    net_radiation = net_shortwave - net_longwave
    wind_speed_2m_m_s = max(wind_speed_2m_m_s, 0.5)
    et0 = (
        0.408 * slope * net_radiation
        + psychrometric_constant * 900.0 / (mean_temperature + 273.0)
        * wind_speed_2m_m_s * (saturation_pressure - actual_vapor_pressure_kpa)
    ) / (slope + psychrometric_constant * (1 + 0.34 * wind_speed_2m_m_s))
    return max(et0, 0.0)


def reference_evapotranspiration_hargreaves(
    tmax_c: float, tmin_c: float, latitude_deg: float, day_of_year: int
) -> float:
    """Temperature-only Hargreaves ET0 fallback in mm/day."""
    radiation_toa = extraterrestrial_radiation(latitude_deg, day_of_year)
    return max(
        0.0023 * ((tmax_c + tmin_c) / 2 + 17.8)
        * math.sqrt(max(tmax_c - tmin_c, 0.0)) * radiation_toa * 0.408,
        0.0,
    )


def crop_coefficient_from_gdd(cumulative_gdd: float, crop: CropParameters = SUGARCANE) -> float:
    """FAO-56 four-stage crop coefficient curve driven by cumulative GDD."""
    if cumulative_gdd <= crop.gdd_initial:
        return crop.kc_initial
    if cumulative_gdd <= crop.gdd_development:
        fraction = (cumulative_gdd - crop.gdd_initial) / (crop.gdd_development - crop.gdd_initial)
        return crop.kc_initial + fraction * (crop.kc_midseason - crop.kc_initial)
    if cumulative_gdd <= crop.gdd_midseason:
        return crop.kc_midseason
    if cumulative_gdd <= crop.gdd_endseason:
        fraction = (cumulative_gdd - crop.gdd_midseason) / (crop.gdd_endseason - crop.gdd_midseason)
        return crop.kc_midseason - fraction * (crop.kc_midseason - crop.kc_endseason)
    return crop.kc_endseason


def crop_coefficient_from_ndvi(ndvi: float, crop: CropParameters = SUGARCANE) -> float:
    """Scale Kc between initial and midseason values using fractional canopy cover."""
    fractional_cover = min(1.0, max(0.0, (ndvi - NDVI_SOIL) / (NDVI_FULL - NDVI_SOIL)))
    return crop.kc_initial + fractional_cover * (crop.kc_midseason - crop.kc_initial)


def root_depth_m(cumulative_gdd: float, crop: CropParameters = SUGARCANE) -> float:
    """Smooth root growth from minimum to maximum depth during crop development."""
    fraction = min(1.0, max(0.0, cumulative_gdd / crop.gdd_development))
    smooth_fraction = 3 * fraction**2 - 2 * fraction**3
    return crop.root_depth_min_m + (crop.root_depth_max_m - crop.root_depth_min_m) * smooth_fraction


def runoff_scs_curve_number(
    rain_mm: float, depletion_fraction: float, curve_number: float = SCS_CURVE_NUMBER
) -> float:
    """Daily SCS curve-number runoff estimate in mm."""
    if rain_mm <= 0:
        return 0.0
    if depletion_fraction < 0.25:
        adjusted_curve_number = 23 * curve_number / (10 + 0.13 * curve_number)
    elif depletion_fraction > 0.75:
        adjusted_curve_number = 4.2 * curve_number / (10 - 0.058 * curve_number)
    else:
        adjusted_curve_number = curve_number
    retention = 25400.0 / adjusted_curve_number - 254.0
    initial_abstraction = 0.2 * retention
    if rain_mm <= initial_abstraction:
        return 0.0
    return (rain_mm - initial_abstraction) ** 2 / (rain_mm + 0.8 * retention)


def root_zone_water_balance(
    depletion_before_mm: float,
    et0_mm_day: float,
    crop_coefficient: float,
    root_depth: float,
    rain_mm: float,
    irrigation_net_mm: float,
    crop: CropParameters = SUGARCANE,
) -> dict[str, float]:
    """Advance one day of FAO-56 root-zone depletion and return its components."""
    total_available_water = SOIL_TAW_MM_PER_M * root_depth
    reference_crop_et = crop_coefficient * et0_mm_day
    depletion_fraction = min(0.8, max(0.1, crop.depletion_fraction + 0.04 * (5.0 - reference_crop_et)))
    readily_available_water = depletion_fraction * total_available_water
    depletion_before_mm = min(max(depletion_before_mm, 0.0), total_available_water)
    if depletion_before_mm <= readily_available_water:
        stress_coefficient = 1.0
    elif depletion_before_mm < total_available_water:
        stress_coefficient = (total_available_water - depletion_before_mm) / (
            (1 - depletion_fraction) * total_available_water
        )
    else:
        stress_coefficient = 0.0
    runoff = runoff_scs_curve_number(rain_mm, depletion_before_mm / total_available_water)
    actual_crop_et = stress_coefficient * reference_crop_et
    unbounded_depletion = depletion_before_mm - (rain_mm - runoff) - irrigation_net_mm + actual_crop_et
    deep_percolation = max(0.0, -unbounded_depletion)
    depletion_after_mm = min(max(unbounded_depletion, 0.0), total_available_water)
    return {
        "total_available_water_mm": total_available_water,
        "readily_available_water_mm": readily_available_water,
        "depletion_fraction": depletion_fraction,
        "stress_coefficient": stress_coefficient,
        "actual_crop_et_mm_day": actual_crop_et,
        "reference_crop_et_mm_day": reference_crop_et,
        "runoff_mm": runoff,
        "deep_percolation_mm": deep_percolation,
        "depletion_after_mm": depletion_after_mm,
    }


def initial_crop_state(
    depletion_fraction: float = 0.30,
    cumulative_gdd: float = 0.0,
    crop: CropParameters = SUGARCANE,
) -> CropState:
    """Create a starting state with depletion set as a fraction of current TAW."""
    if not 0.0 <= depletion_fraction <= 1.0:
        raise ValueError("Initial depletion fraction must be between 0 and 1")
    total_available_water = SOIL_TAW_MM_PER_M * root_depth_m(cumulative_gdd, crop)
    return CropState(cumulative_gdd, depletion_fraction * total_available_water)


def assess_irrigation_need(
    farm: FarmConfig,
    weather: DailyWeather,
    state: CropState,
    *,
    ndvi: float | None = None,
    ndvi_weight: float = 1.0,
    irrigation_applied_gross_mm: float = 0.0,
    rain_mm_override: float | None = None,
    recent_reference_crop_et_mm_day: Sequence[float] = (),
    crop: CropParameters = SUGARCANE,
) -> IrrigationAdvice:
    """Calculate today's irrigation status and return the updated crop state.

    Supply measured weather and the prior day's cumulative GDD/depletion. NDVI is
    optional; pass ``ndvi_weight`` from 0 (ignore) to 1 (fully trust) when its
    observation is not from today. Irrigation input is gross applied depth.
    """
    if farm.area_m2 <= 0:
        raise ValueError("Farm area must be greater than zero")
    if weather.date < farm.planting_date:
        raise ValueError("Weather date cannot be before the crop planting date")
    if state.cumulative_gdd < 0 or state.depletion_mm < 0:
        raise ValueError("Crop state values cannot be negative")
    if state.last_processed_date is not None and weather.date != state.last_processed_date + dt.timedelta(days=1):
        raise ValueError("Weather must be for the day immediately after the crop state's last processed date")
    if not 0.0 <= ndvi_weight <= 1.0:
        raise ValueError("NDVI weight must be between 0 and 1")
    if ndvi is not None and not -1.0 <= ndvi <= 1.0:
        raise ValueError("NDVI must be between -1 and 1")
    if irrigation_applied_gross_mm < 0 or (rain_mm_override is not None and rain_mm_override < 0):
        raise ValueError("Rainfall and irrigation amounts cannot be negative")

    day_of_year = weather.date.timetuple().tm_yday
    gdd_today = growing_degree_days(weather.tmax_c, weather.tmin_c, crop)
    cumulative_gdd = state.cumulative_gdd + gdd_today
    kc_gdd = crop_coefficient_from_gdd(cumulative_gdd, crop)
    if ndvi is None:
        crop_coefficient = kc_gdd
        coefficient_source = "gdd"
    else:
        kc_ndvi = crop_coefficient_from_ndvi(ndvi, crop)
        crop_coefficient = ndvi_weight * kc_ndvi + (1 - ndvi_weight) * kc_gdd
        coefficient_source = "ndvi" if ndvi_weight >= 0.99 else "blend" if ndvi_weight > 0.01 else "gdd"

    meteorological_values = (
        weather.actual_vapor_pressure_kpa,
        weather.solar_radiation_mj_m2,
        weather.wind_speed_2m_m_s,
    )
    if all(value is not None for value in meteorological_values):
        et0 = reference_evapotranspiration_penman_monteith(
            weather.tmax_c,
            weather.tmin_c,
            weather.actual_vapor_pressure_kpa,
            weather.solar_radiation_mj_m2,
            weather.wind_speed_2m_m_s,
            farm.elevation_m,
            farm.latitude_deg,
            day_of_year,
        )
        et0_method = "penman_monteith"
    else:
        et0 = reference_evapotranspiration_hargreaves(
            weather.tmax_c, weather.tmin_c, farm.latitude_deg, day_of_year
        )
        et0_method = "hargreaves"

    root_depth = root_depth_m(cumulative_gdd, crop)
    rain = weather.rain_mm if rain_mm_override is None else rain_mm_override
    irrigation_net = irrigation_applied_gross_mm * IRRIGATION_EFFICIENCY
    balance = root_zone_water_balance(
        state.depletion_mm, et0, crop_coefficient, root_depth, rain, irrigation_net, crop
    )
    depletion = balance["depletion_after_mm"]
    trigger = min(MAD_TRIGGER * balance["total_available_water_mm"], balance["readily_available_water_mm"])
    reference_crop_et = crop_coefficient * et0
    irrigate_now = depletion + LEAD_DAYS * reference_crop_et >= trigger

    if irrigate_now:
        recommended_net = min(depletion, MAX_NET_EVENT_MM)
        recommended_gross = recommended_net / IRRIGATION_EFFICIENCY
        recommended_liters = recommended_gross * farm.area_m2
        application_rate = farm.drip_application_rate_mm_per_hour
        recommended_hours = recommended_gross / application_rate if application_rate > 0 else 0.0
        days_to_trigger = None
    else:
        recommended_net = recommended_gross = recommended_liters = recommended_hours = 0.0
        average_crop_et = (
            sum(recent_reference_crop_et_mm_day) / len(recent_reference_crop_et_mm_day)
            if recent_reference_crop_et_mm_day
            else reference_crop_et
        )
        days_to_trigger = max(1, math.ceil((trigger - depletion) / average_crop_et)) if average_crop_et > 0.2 else None

    return IrrigationAdvice(
        farm_id=farm.farm_id,
        date=weather.date,
        status="IRRIGATE_NOW" if irrigate_now else "NO_IRRIGATION_NEEDED",
        irrigate_now=irrigate_now,
        recommended_net_mm=recommended_net,
        recommended_gross_mm=recommended_gross,
        recommended_liters=recommended_liters,
        recommended_drip_hours=recommended_hours,
        depletion_mm=depletion,
        trigger_mm=trigger,
        days_to_trigger=days_to_trigger,
        et0_mm_day=et0,
        et0_method=et0_method,
        crop_coefficient=crop_coefficient,
        crop_coefficient_source=coefficient_source,
        root_depth_m=root_depth,
        crop_stress_coefficient=balance["stress_coefficient"],
        runoff_mm=balance["runoff_mm"],
        deep_percolation_mm=balance["deep_percolation_mm"],
        next_state=CropState(cumulative_gdd, depletion, weather.date),
    )