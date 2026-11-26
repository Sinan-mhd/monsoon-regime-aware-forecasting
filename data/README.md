# Data Directory

This repository does not include real MoES or IMD meteorological observations or NWP forecasts.

- Running `python train.py` without `--data` looks for `data/monsoon_data.csv`; when absent, it generates `data/demo_monsoon.csv` (clearly labelled synthetic demonstration data).
- For real experiments, place your quality-controlled CSV at `data/monsoon_data.csv` (or pass `--data <path>`) following the schema documented in the root `README.md`, including an optional `date` column (`YYYY-MM-DD`) to enable chronological splitting.
- Document dataset source, spatial/temporal coverage, and regime-labelling methodology before committing or reporting results.
