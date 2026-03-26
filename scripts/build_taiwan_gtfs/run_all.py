"""
Convenience runner: execute all 4 build steps in order.

Usage:
  export GOOGLE_MAPS_API_KEY="AIza..."
  python3 scripts/build_taiwan_gtfs/run_all.py

Steps 1 (Wikipedia scrape) and 4 (offline GTFS build) need no API key.
Steps 2 (geocode) and 3 (fetch schedules) require GOOGLE_MAPS_API_KEY.

You can run individual steps if you need to re-run just part of the pipeline:
  python3 scripts/build_taiwan_gtfs/scrape_stations.py
  python3 scripts/build_taiwan_gtfs/geocode_stations.py
  python3 scripts/build_taiwan_gtfs/fetch_schedules.py
  python3 scripts/build_taiwan_gtfs/build_gtfs.py
"""

import sys
import pathlib
import importlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

STEPS = [
    ("scrape_stations",  "Step 1: Scrape station names from Wikipedia"),
    ("geocode_stations", "Step 2: Geocode stations via Google Maps Geocoding API"),
    ("fetch_schedules",  "Step 3: Fetch departure schedules via Google Maps Routes API"),
    ("build_gtfs",       "Step 4: Build GTFS files (offline)"),
]


def run_step(module_name: str, description: str):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    mod = importlib.import_module(module_name)
    mod.main()


if __name__ == "__main__":
    # Allow skipping to a particular step: python run_all.py 3
    start_step = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    for i, (module, description) in enumerate(STEPS, start=1):
        if i < start_step:
            print(f"Skipping {description}")
            continue
        run_step(module, description)

    print("\n\nAll steps complete.")
    print("Next: run  python3 verify_network.py  to confirm the feed loads correctly.")
    print("Then: update config.yaml to point at data/tra_gtfs and data/thsr_gtfs.")
