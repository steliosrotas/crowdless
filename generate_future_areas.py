import json
from pathlib import Path

FUTURE_DIR = Path("./future_areas")
FUTURE_DIR.mkdir(exist_ok=True)

AREA_COORDS = {
    "Nea_Smyrni": [23.7190, 37.9365, 23.7400, 37.9530],
    "Kallithea": [23.7110, 37.9550, 23.7360, 37.9750],
    "Pireaus": [23.6290, 37.9370, 23.6640, 37.9640],
    "Zografou": [23.7740, 37.9770, 23.8080, 38.0020],
    "Syntagma": [23.7410, 37.9715, 23.7475, 37.9765],
    "Eksarhia": [23.7335, 37.9840, 23.7405, 37.9905],
    "Pagkrati": [23.7495, 37.9635, 23.7605, 37.9735],
    "Aghios_Dimitrios": [23.7380, 37.9445, 23.7555, 37.9590],
}

# required datetime for all generated items
FUTURE_DATETIME = "2026-11-03T09:20:07.144Z"

for area, coords in AREA_COORDS.items():
    payload = {
        "area": area,
        "coords": coords,
        "start": FUTURE_DATETIME,
        "end": FUTURE_DATETIME,
        "count": 1,
        "items": [
            {
                "id": f"future_{area}_sample",
                "datetime": FUTURE_DATETIME,
                "cloud_cover": None,
                "platform": "simulated",
                "bbox": coords,
                # minimal placeholders for reference; real metrics can be filled later
                "ndvi_mean": None,
                "ndbi_mean": None,
                "brightness_mean": None,
            }
        ],
    }

    outpath = FUTURE_DIR / f"{area}.json"
    with outpath.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

print(f"Wrote {len(AREA_COORDS)} files to {FUTURE_DIR.resolve()}")