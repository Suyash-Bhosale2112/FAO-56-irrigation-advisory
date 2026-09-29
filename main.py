"""Command-line entry point for the farm-level irrigation backend."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from irrigation_backend import GEEDataProvider, advise_farm_with_gee
from irrigation_core import CropState, FarmConfig, initial_crop_state


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FARM_CONFIG = PROJECT_DIR / "farms.json"
DEFAULT_STATE_FILE = PROJECT_DIR / "state.json"
DEFAULT_LOG_FILE = PROJECT_DIR / "logs" / "advisor.log"


def configure_logging() -> None:
    log_path = Path(os.getenv("LOG_PATH", str(DEFAULT_LOG_FILE)))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )


def _required_number(data: dict, key: str, farm_id: str) -> float:
    value = data.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{farm_id}: '{key}' must be a number in farms.json") from error
    if not math.isfinite(number):
        raise ValueError(f"{farm_id}: '{key}' must be finite")
    return number


def load_farms(config_path: str | Path = DEFAULT_FARM_CONFIG) -> list[FarmConfig]:
    """Read and validate farm locations and irrigation-system configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Farm configuration not found: {path}. Copy farms.example.json to farms.json and fill in real farm values."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    farms_data = data.get("farms")
    if not isinstance(farms_data, list) or not farms_data:
        raise ValueError("farms.json must contain a non-empty 'farms' list")

    farms = []
    farm_ids = set()
    for item in farms_data:
        farm_id = str(item.get("farm_id", "")).strip()
        if not farm_id:
            raise ValueError("Every farm needs a non-empty 'farm_id'")
        if farm_id in farm_ids:
            raise ValueError(f"Duplicate farm_id in farms.json: {farm_id}")
        farm_ids.add(farm_id)
        latitude = _required_number(item, "latitude_deg", farm_id)
        longitude = _required_number(item, "longitude_deg", farm_id)
        area = _required_number(item, "area_m2", farm_id)
        elevation = _required_number(item, "elevation_m", farm_id)
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"{farm_id}: latitude/longitude are outside valid ranges")
        if area <= 0 or elevation < -500 or elevation > 10000:
            raise ValueError(f"{farm_id}: area must be positive and elevation must be between -500 and 10000 m")
        try:
            planting_date = dt.date.fromisoformat(item["planting_date"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{farm_id}: 'planting_date' must use YYYY-MM-DD format") from error
        flow_lph = _required_number(item, "flow_lph", farm_id)
        emitters = _required_number(item, "emitters_per_acre", farm_id)
        if flow_lph <= 0 or emitters <= 0 or not emitters.is_integer():
            raise ValueError(f"{farm_id}: flow_lph and positive integer emitters_per_acre are required")
        farms.append(
            FarmConfig(
                farm_id=farm_id,
                latitude_deg=latitude,
                longitude_deg=longitude,
                area_m2=area,
                planting_date=planting_date,
                flow_lph=flow_lph,
                emitters_per_acre=int(emitters),
                elevation_m=elevation,
                geometry_geojson=item.get("geometry_geojson"),
            )
        )
    return farms


def load_states(state_path: str | Path) -> dict[str, CropState]:
    path = Path(state_path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    states = {}
    for farm_id, data in raw.items():
        processed_date = data.get("last_processed_date")
        states[farm_id] = CropState(
            cumulative_gdd=float(data["cumulative_gdd"]),
            depletion_mm=float(data["depletion_mm"]),
            last_processed_date=dt.date.fromisoformat(processed_date) if processed_date else None,
        )
    return states


def save_states(state_path: str | Path, states: dict[str, CropState]) -> None:
    """Atomically persist crop states so an interrupted run does not corrupt them."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        farm_id: {
            "cumulative_gdd": state.cumulative_gdd,
            "depletion_mm": state.depletion_mm,
            "last_processed_date": state.last_processed_date.isoformat() if state.last_processed_date else None,
        }
        for farm_id, state in states.items()
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def format_advice(advice) -> str:
    if advice.irrigate_now:
        action = (
            f"Apply {advice.recommended_gross_mm:.1f} mm gross "
            f"({advice.recommended_liters:,.0f} L); run drip {advice.recommended_drip_hours:.2f} h"
        )
    else:
        next_irrigation = (
            f"; estimated {advice.days_to_trigger} day(s) to trigger"
            if advice.days_to_trigger is not None
            else ""
        )
        action = "No irrigation recommended" + next_irrigation
    return (
        f"{advice.farm_id} | {advice.date} | {advice.status}\n"
        f"  {action}\n"
        f"  Depletion {advice.depletion_mm:.1f}/{advice.trigger_mm:.1f} mm; "
        f"ET0 {advice.et0_mm_day:.2f} mm/day ({advice.et0_method}); "
        f"Kc {advice.crop_coefficient:.2f} ({advice.crop_coefficient_source})"
    )


def run_advisor(
    farm_config_path: str | Path = DEFAULT_FARM_CONFIG,
    state_path: str | Path = DEFAULT_STATE_FILE,
    *,
    provider=None,
) -> int:
    farms = load_farms(farm_config_path)
    states = load_states(state_path)
    provider = provider or GEEDataProvider.from_environment()
    image_count = provider.verify_access()
    logging.info("Earth Engine access verified; ERA5-Land image count: %s", image_count)

    failures = 0
    for farm in farms:
        previous_state = states.get(farm.farm_id)
        if previous_state is None:
            previous_state = initial_crop_state()
        try:
            advice = advise_farm_with_gee(farm, previous_state, provider=provider)
        except Exception as error:
            failures += 1
            logging.exception("Advice calculation failed for %s", farm.farm_id)
            print(f"{farm.farm_id} | ERROR | {error}", file=sys.stderr)
            continue
        states[farm.farm_id] = advice.next_state
        save_states(state_path, states)
        print(format_advice(advice))
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(PROJECT_DIR / ".env")
    key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    if key_path and not Path(key_path).is_absolute():
        os.environ["GEE_SERVICE_ACCOUNT_KEY"] = str((PROJECT_DIR / key_path).resolve())
    parser = argparse.ArgumentParser(description="Fetch GEE observations and calculate farm irrigation advice")
    parser.add_argument("--config", default=str(DEFAULT_FARM_CONFIG), help="JSON farm configuration path")
    parser.add_argument("--state", default=str(DEFAULT_STATE_FILE), help="JSON crop-state persistence path")
    parser.add_argument(
        "--check-earth-engine",
        action="store_true",
        help="verify credentials and ERA5-Land catalog access without calculating recommendations",
    )
    args = parser.parse_args(argv)
    configure_logging()
    try:
        if args.check_earth_engine:
            provider = GEEDataProvider.from_environment()
            image_count = provider.verify_access()
            print(f"Earth Engine access verified; ERA5-Land image count: {image_count}")
            return 0
        return run_advisor(args.config, args.state)
    except Exception as error:
        logging.exception("Irrigation backend could not complete")
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())