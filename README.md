# Crowdless

Minimal pipeline to compute and visualize a **crowdness score** per area over time. Includes a TypeScript UI demo and a Jupyter notebook for data processing, exploration, and scoring.

---

## Contents

- `UI_Demo/` — lightweight web UI demo for browsing areas and inspecting scores  
- `Crowdness_API_analysis_Demo.ipynb` — generic, feature-based scoring notebook  

---

## Data Format

Place per–area JSON files under `areas_output/`:

```json
{
  "area": "Syntagma",
  "items": [
    {
      "datetime": "2025-07-01T13:00:00Z",
      "temperature_celsius": 30.5,
      "relative_humidity_percent": 55.0
    }
  ]
}
```

## UI Demo

Configure the demo’s data source to point to your precomputed outputs or service as needed.

---

## Notebooks

### Generic Feature-Based Scoring
**File:** `Crowdness_API_analysis_Demo.ipynb`

Capabilities:
- Reads `areas_output/*.json`
- Builds a lookup dataframe `areas_output/metrics_database.pkl`
- Saves a simple, tunable scoring model `areas_output/crowdness_model.json`
- Single and batch predictions via nearest-timestamp lookup

### Data Collector Notebook
**File:** `Crowdless_Crowdness_Demo_with_collector.ipynb`

Adds a cell that writes `collect_s2_catalog.py` (Copernicus Data Space, Sentinel-2 L2A), collects scene metadata for predefined areas/date ranges, normalizes items, and saves JSON under `areas_output/` (per-area or per-day). When real processing is off, it emits deterministic fake indices for testing.

Example:
```bash
python collect_s2_catalog.py --areas Syntagma,Kallithea --start 2025-01-01T00:00:00Z --end 2025-01-07T23:59:59Z --cloud-max 30 --out areas_output

```
Place JSON data under `areas_output/`, open the notebook(s), and run cells top-to-bottom. Edit `FIELD_MAP` in the generic notebook to match your numeric fields.

```

---

## Scoring Model (Notebook)

Interpretable, rule-based score:

- `feature_x`: triangular peak at `ideal_x`, falling to 0 at `ideal_x ± range_x`
- `feature_y`: full score at or below `ideal_y`, linear penalty above up to 0 at `ideal_y + range_y`
- Final score: weighted average of the two sub-scores in `[0, 100]`

Tune `crowdness_model.json` parameters to fit your use case.

---
