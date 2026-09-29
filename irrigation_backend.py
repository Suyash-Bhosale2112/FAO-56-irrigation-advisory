"""Google Earth Engine data access and daily farm-level irrigation workflow."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path

from irrigation_core import (
    CropParameters,
    CropState,
    DailyWeather,
    FarmConfig,
    IrrigationAdvice,
    SUGARCANE,
    assess_irrigation_need,
    saturation_vapor_pressure,
)


class WeatherDataUnavailableError(RuntimeError):
    """Raised when Earth Engine cannot supply a contiguous weather history."""


class GEEDataProvider:
    """Fetch ERA5-Land daily weather and Sentinel-2 NDVI from Earth Engine."""

    WEATHER_BANDS = (
        "temperature_2m_max",
        "temperature_2m_min",
        "dewpoint_temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards_sum",
        "total_precipitation_sum",
    )

    def __init__(self, service_account_key: str | Path, project_id: str | None = None):
        try:
            import ee
        except ImportError as error:
            raise RuntimeError("Install dependencies with 'pip install -r requirements.txt'") from error

        key_path = Path(service_account_key)
        if not key_path.is_file():
            raise FileNotFoundError(f"Earth Engine service-account key not found: {key_path}")
        key_data = json.loads(key_path.read_text(encoding="utf-8"))
        credentials = ee.ServiceAccountCredentials(key_data["client_email"], str(key_path))
        project = project_id or key_data.get("project_id")
        if project:
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(credentials)
        self.ee = ee

    @classmethod
    def from_environment(cls) -> GEEDataProvider:
        key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
        if not key_path:
            raise RuntimeError("Set GEE_SERVICE_ACCOUNT_KEY to the Earth Engine service-account JSON file")
        return cls(key_path, os.getenv("GEE_PROJECT_ID"))

    def _farm_geometry(self, farm: FarmConfig):
        if farm.geometry_geojson is not None:
            return self.ee.Geometry(farm.geometry_geojson)
        return self.ee.Geometry.Point([farm.longitude_deg, farm.latitude_deg])

    def verify_access(self) -> int:
        """Verify Earth Engine credentials and access to the ERA5-Land collection."""
        image_count = (
            self.ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .limit(1)
            .size()
            .getInfo()
        )
        if not image_count:
            raise RuntimeError("Earth Engine connected, but the ERA5-Land collection returned no images")
        return image_count

    def weather_series(
        self, farm: FarmConfig, start: dt.date, end: dt.date
    ) -> dict[dt.date, DailyWeather]:
        """Fetch daily ERA5-Land at the farm coordinate; ERA5-Land is about 11 km resolution."""
        ee = self.ee
        point = ee.Geometry.Point([farm.longitude_deg, farm.latitude_deg])
        collection = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat())
            .select(list(self.WEATHER_BANDS))
        )

        def to_feature(image):
            values = image.reduceRegion(ee.Reducer.first(), point, scale=11132, maxPixels=1000000)
            return ee.Feature(None, values).set("date", image.date().format("YYYY-MM-dd"))

        result = collection.map(to_feature).getInfo()
        weather_by_date = {}
        for feature in result.get("features", []):
            properties = feature["properties"]
            if properties.get("temperature_2m_max") is None or properties.get("temperature_2m_min") is None:
                continue
            dewpoint_kelvin = properties.get("dewpoint_temperature_2m")
            wind_u = properties.get("u_component_of_wind_10m")
            wind_v = properties.get("v_component_of_wind_10m")
            solar_radiation = properties.get("surface_solar_radiation_downwards_sum")
            wind_speed_2m = None
            if wind_u is not None and wind_v is not None:
                wind_speed_10m = math.hypot(wind_u, wind_v)
                wind_speed_2m = wind_speed_10m * 4.87 / math.log(67.8 * 10 - 5.42)
            day = dt.date.fromisoformat(properties["date"])
            weather_by_date[day] = DailyWeather(
                date=day,
                tmax_c=properties["temperature_2m_max"] - 273.15,
                tmin_c=properties["temperature_2m_min"] - 273.15,
                rain_mm=max(0.0, (properties.get("total_precipitation_sum") or 0.0) * 1000.0),
                actual_vapor_pressure_kpa=(
                    saturation_vapor_pressure(dewpoint_kelvin - 273.15)
                    if dewpoint_kelvin is not None
                    else None
                ),
                solar_radiation_mj_m2=solar_radiation / 1e6 if solar_radiation is not None else None,
                wind_speed_2m_m_s=wind_speed_2m,
            )
        return weather_by_date

    def ndvi_series(
        self, farm: FarmConfig, start: dt.date, end: dt.date
    ) -> dict[dt.date, float]:
        """Fetch clear Sentinel-2 surface-reflectance NDVI over the farm geometry."""
        ee = self.ee
        geometry = self._farm_geometry(farm)
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat())
        )

        def image_ndvi(image):
            classification = image.select("SCL")
            valid = classification.eq(4).Or(classification.eq(5)).unmask(0).rename("valid")
            ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI").updateMask(valid)
            stats = ndvi.addBands(valid).reduceRegion(
                ee.Reducer.mean(), geometry, scale=10, maxPixels=1000000
            )
            return ee.Feature(None, stats).set("date", image.date().format("YYYY-MM-dd"))

        result = collection.map(image_ndvi).getInfo()
        by_day: dict[dt.date, list[float]] = {}
        for feature in result.get("features", []):
            properties = feature["properties"]
            value = properties.get("NDVI")
            if value is not None and (properties.get("valid") or 0.0) >= 0.5 and value > 0.05:
                day = dt.date.fromisoformat(properties["date"])
                by_day.setdefault(day, []).append(value)
        return {day: sum(values) / len(values) for day, values in by_day.items()}


def _ndvi_for_day(observations: dict[dt.date, float], day: dt.date) -> tuple[float | None, float]:
    if not observations:
        return None, 0.0
    days = sorted(observations)
    previous = [observation_day for observation_day in days if observation_day <= day]
    following = [observation_day for observation_day in days if observation_day >= day]
    if previous and following:
        before, after = previous[-1], following[0]
        value = observations[before] if before == after else observations[before] + (
            observations[after] - observations[before]
        ) * (day - before).days / (after - before).days
        distance = min((day - before).days, (after - day).days)
    elif previous:
        before = previous[-1]
        value, distance = observations[before], (day - before).days
    else:
        after = following[0]
        value, distance = observations[after], (after - day).days
    trust_weight = 1.0 if distance <= 5 else max(0.0, 1.0 - (distance - 5) / 20.0)
    return value, trust_weight


def advise_farm_with_gee(
    farm: FarmConfig,
    state: CropState,
    *,
    provider: GEEDataProvider | None = None,
    target_date: dt.date | None = None,
    irrigation_by_date_gross_mm: dict[dt.date, float] | None = None,
    rain_by_date_mm: dict[dt.date, float] | None = None,
    crop: CropParameters = SUGARCANE,
) -> IrrigationAdvice:
    """Fetch GEE observations, catch the crop state up, and return latest advice.

    When state has no processed date, weather is fetched from planting day so crop
    stage is accumulated from the actual planting date. The caller persists
    ``advice.next_state`` and supplies it on the next invocation.
    """
    provider = provider or GEEDataProvider.from_environment()
    target_date = target_date or dt.date.today()
    start_date = state.last_processed_date + dt.timedelta(days=1) if state.last_processed_date else farm.planting_date
    if start_date > target_date:
        raise ValueError("Crop state is already processed through the requested date")

    weather_by_date = provider.weather_series(farm, start_date, target_date)
    if not weather_by_date:
        raise WeatherDataUnavailableError(
            f"No ERA5-Land weather is available from {start_date} through {target_date}; "
            "no irrigation status was calculated."
        )
    observed_days = sorted(weather_by_date)
    expected_days = (observed_days[-1] - start_date).days + 1
    if observed_days[0] != start_date or len(observed_days) != expected_days:
        missing = [start_date + dt.timedelta(days=offset) for offset in range(expected_days)
                   if start_date + dt.timedelta(days=offset) not in weather_by_date]
        raise WeatherDataUnavailableError(
            f"ERA5-Land data has missing days in the required state history: {missing[:5]}"
        )

    ndvi_observations = provider.ndvi_series(
        farm, max(farm.planting_date, start_date - dt.timedelta(days=30)), observed_days[-1]
    )
    current_state = state
    recent_reference_crop_et: list[float] = []
    advice = None
    irrigation_by_date_gross_mm = irrigation_by_date_gross_mm or {}
    rain_by_date_mm = rain_by_date_mm or {}
    for day in observed_days:
        ndvi, ndvi_weight = _ndvi_for_day(ndvi_observations, day)
        advice = assess_irrigation_need(
            farm,
            weather_by_date[day],
            current_state,
            ndvi=ndvi,
            ndvi_weight=ndvi_weight,
            irrigation_applied_gross_mm=irrigation_by_date_gross_mm.get(day, 0.0),
            rain_mm_override=rain_by_date_mm.get(day),
            recent_reference_crop_et_mm_day=recent_reference_crop_et[-7:],
            crop=crop,
        )
        current_state = advice.next_state
        recent_reference_crop_et.append(advice.crop_coefficient * advice.et0_mm_day)

    if advice is None:
        raise WeatherDataUnavailableError("Earth Engine returned no usable daily weather observations")
    return advice