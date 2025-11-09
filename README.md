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
