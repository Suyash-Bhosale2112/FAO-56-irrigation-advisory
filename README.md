# Farm-Level Irrigation Backend

This is a backend-only sugarcane irrigation advisor. It fetches weather from Google Earth Engine's ERA5-Land daily collection and vegetation observations from Sentinel-2, then runs the FAO-56 water-balance algorithm. There is no Telegram bot, web frontend, scheduler, database, dry-run mode, or bundled test harness.

## Data Flow

1. The caller supplies each farm's id, latitude/longitude, area, planting date, and drip-system details.
2. `GEEDataProvider` fetches ERA5-Land daily weather at the farm coordinate. Sentinel-2 NDVI is averaged over `geometry_geojson` when supplied, otherwise at the farm coordinate.
3. `advise_farm_with_gee()` converts the observations, interpolates and age-weights NDVI, then advances the crop water state day-by-day from planting or the last saved state through the latest available ERA5-Land date.
4. The function returns the latest available day's status, water recommendation, and `next_state`. The calling system must persist that state and provide it on the next call.

ERA5-Land is approximately 11 km resolution, so weather represents the regional pixel rather than farm-specific microclimate. Sentinel-2 observations are cloud-screened and may be sparse. No status is returned when required weather days are missing; the workflow raises `WeatherDataUnavailableError` rather than filling gaps with fabricated observations.

## Setup

Python 3.10 or newer and an enabled Google Earth Engine project/service account are required. Install dependencies and create a farm configuration from the blank template:

```powershell
python -m pip install -r requirements.txt
Copy-Item farms.example.json farms.json
```

Edit `farms.json` with real farm coordinates, area, planting date, elevation, and drip-system values. Add a GeoJSON Polygon to `geometry_geojson` if Sentinel-2 NDVI should be averaged over the farm boundary; `null` uses the farm coordinate. This local farm file is ignored by Git.

The local `.env` points `GEE_SERVICE_ACCOUNT_KEY` to the service-account JSON and provides `GEE_PROJECT_ID`. The service account must be registered for Earth Engine use and have access to the project.

## Daily Backend Call

```python
import datetime as dt

from irrigation_backend import advise_farm_with_gee
from irrigation_core import FarmConfig, initial_crop_state

farm = FarmConfig(
    farm_id="FARM_1",
    latitude_deg=17.6,
    longitude_deg=74.2,
    area_m2=10_117,
    planting_date=dt.date(2025, 8, 1),
    flow_lph=4.0,
    emitters_per_acre=4000,
    elevation_m=600.0,
)

# Load the saved CropState on later calls; initialize only for the first call.
state = initial_crop_state()
advice = advise_farm_with_gee(farm, state)
print(advice.date, advice.status)
print(advice.recommended_gross_mm, advice.recommended_liters, advice.recommended_drip_hours)

# Persist advice.next_state for the next daily call.
state = advice.next_state
```

Run the cloud-access check after replacing the exposed key, then run the daily calculation:

```powershell
python main.py --check-earth-engine
python main.py
```

`main.py` loads `.env`, validates the farm file, checks access to the ERA5-Land catalog, fetches observations, prints advice, and saves each farm's crop state in the ignored `state.json`. The first run downloads daily weather from planting date to the latest available date to initialize cumulative GDD and soil depletion. Later runs fetch any missing days and process them in order. Weather latency is possible, so the advice date is the date actually analyzed, not necessarily today's date. Run `python main.py --help` for configuration/state file options. For automatic daily operation, schedule `python main.py` with Windows Task Scheduler.

`status` is `IRRIGATE_NOW` or `NO_IRRIGATION_NEEDED`. The result includes ET0 method, crop coefficient/source, root depth, soil depletion and trigger, runoff, deep percolation, estimated liters, and drip hours. Irrigation and rain measurements can be supplied as date-keyed mappings to `advise_farm_with_gee()` so they are included during catch-up.

The calculation and parameters are in `irrigation_core.py`; GEE access and daily orchestration are in `irrigation_backend.py`. Calibrate the crop-stage parameters, soil available water, curve number, and drip-system values for the actual farm before relying on recommendations.
# FAO-56-irrigation-advisory
