import os
import argparse
import requests
import json
import time
import numpy as np
# import imageio.v2 as imageio # Not needed if not computing real indices
from datetime import datetime, timedelta
import os
import argparse
import random
import hashlib

# === CONFIGURATION ===
# (All your configuration remains the same)
CLIENT_ID = "sh-f6624d72-4c2e-4d3c-bf5e-97d492c49d07"
CLIENT_SECRET = "j59rYz9b06Op9vPnA7FtSXwsbmVT400T"
DEFAULT_BBOX = [23.70, 37.95, 23.78, 38.02]
DEFAULT_START = "2016-01-01T00:00:00Z"
DEFAULT_END = "2025-11-30T23:59:59Z"
AREAS = {
    "Kallithea": [23.7110, 37.9550, 23.7360, 37.9750],
    "Pireaus": [23.6290, 37.9370, 23.6640, 37.9640],
    "Zografou": [23.7740, 37.9770, 23.8080, 38.0020],
    "Syntagma": [23.7410, 37.9715, 23.7475, 37.9765],
    "Eksarhia": [23.7335, 37.9840, 23.7405, 37.9905],
    "Pagkrati": [23.7495, 37.9635, 23.7605, 37.9735],
    "Nea_Smyrni": [23.7190, 37.9365, 23.7400, 37.9530],
    "Aghios_Dimitrios": [23.7380, 37.9445, 23.7555, 37.9590],
}
SPECIAL_DATES = set([
    "2025-02-28",
    "2025-05-01", "2024-05-01", "2023-05-01", "2022-05-01", "2019-05-01", "2018-05-01", "2017-05-01", "2016-05-01",
    "2024-12-25", "2023-12-25", "2022-12-25", "2019-12-25", "2018-12-25", "2017-12-25", "2016-12-25",
])
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
LEGACY_SH_CATALOG = "https://services.sentinel-hub.com/api/v1/catalog/search"

# --- HELPER FUNCTIONS ---
# (I have removed the duplicated functions from here)

def ensure_credentials():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise SystemExit("Client ID and secret must be set")

def get_access_token(client_id, client_secret, timeout=30):
    # (This function is defined once here)
    if not client_id or not client_secret:
        raise SystemExit("Missing client id/secret. Set CDS_CLIENT_ID and CDS_CLIENT_SECRET environment variables.")
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    r = requests.post(TOKEN_URL, data=data, timeout=timeout)
    if r.status_code == 401:
        raise SystemExit("Unauthorized: check client id/secret and that they are for Copernicus Data Space (401).")
    try:
        r.raise_for_status()
    except requests.HTTPError:
        raise SystemExit(f"Failed to fetch token: {r.status_code} {r.text}")
    return r.json().get("access_token")

def search_catalog_page(token, payload, max_retries=5, backoff_base=0.5, timeout=60):
    # (This function is defined once here)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.post(CATALOG_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            if attempt >= max_retries:
                raise SystemExit(f"Catalog request failed after {attempt} attempts: {e}")
            sleep = backoff_base * (2 ** (attempt - 1))
            time.sleep(sleep)
            continue
        if r.status_code in (429, 503) and attempt < max_retries:
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    sleep = float(ra)
                except Exception:
                    sleep = backoff_base * (2 ** (attempt - 1))
            else:
                sleep = backoff_base * (2 ** (attempt - 1))
            time.sleep(sleep)
            continue
        if r.status_code >= 400:
            raise requests.HTTPError(f"Catalog search failed: {r.status_code} - {r.text}", response=r)
        return r.json()

def collect_results(token, bbox, start_date, end_date, cloud_max, per_page=50, max_items=500):
    # (This function is defined once here)
    results = []
    base_payload = {
        "bbox": bbox,
        "datetime": f"{start_date}/{end_date}",
        "collections": ["sentinel-2-l2a"],
        "limit": per_page,
    }
    payload_with_filter = dict(base_payload)
    payload_with_filter["filter"] = f"eo:cloud_cover <= {cloud_max}"
    used_filter = False
    resp = None
    for payload in (payload_with_filter, base_payload):
        try:
            resp = search_catalog_page(token, payload)
            used_filter = (payload is payload_with_filter)
            break
        except requests.HTTPError as e:
            if payload is base_payload:
                raise SystemExit(f"Catalog search failed after retries: {e}")
            time.sleep(0.2)
            continue
    if not resp:
        return results
    items = resp.get("features", []) or []
    results.extend(items)
    context = resp.get("context", {}) or {}
    next_token = context.get("next")
    while next_token and len(results) < max_items:
        paged_payload = dict(base_payload)
        if used_filter:
            paged_payload["filter"] = payload_with_filter["filter"]
        paged_payload["next"] = next_token
        try:
            resp = search_catalog_page(token, paged_payload)
        except requests.HTTPError as e:
            raise SystemExit(f"Pagination request failed: {e}")
        items = resp.get("features", []) or []
        results.extend(items)
        context = resp.get("context", {}) or {}
        next_token = context.get("next")
    return results[:max_items]

# *** MODIFIED FUNCTION ***
def _generate_fake_indices(date_iso, scene_id=None):
    """
    Deterministic per-date base plus per-scene jitter.
    Returns (ndvi, ndbi, brightness, temp, humidity).
    Special dates get higher NDVI/NDBI/brightness.
    """
    if not date_iso:
        date_key = "unknown"
        month = 1 # Default month
    else:
        date_key = date_iso.split("T")[0]
        try:
            month = datetime.fromisoformat(date_key).month
        except ValueError:
            month = 1

    base_seed = int(hashlib.sha256(date_key.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    base_rng = random.Random(base_seed)

    # --- Seasonal base ranges for Athens ---
    # Temp (Celsius): Min in Jan (~10C), Max in July (~33C)
    # np.sin(np.pi * (month - 1) / 12) -> not quite right
    # Use simple lookup or cosine wave
    # (month - 1) * (np.pi / 6) -> 0 for Jan, pi/3 for Mar, etc.
    # Center max (1) at July (month=7, idx=6) -> cos( (idx-6) * pi / 6 )
    month_idx = month - 1
    # Base temp: avg 21.5C, amplitude 11.5C (Range ~10C to 33C)
    base_temp = 21.5 - 11.5 * np.cos(np.pi * (month_idx - 6) / 6)
    # Base humidity: Drier in summer. Avg 60%, amplitude 15%
    base_humidity = 60 + 15 * np.cos(np.pi * (month_idx - 6) / 6)
    
    # Add daily random variation
    base_temp += base_rng.uniform(-3, 3)
    base_humidity += base_rng.uniform(-5, 5)

    # choose base ranges for indices
    if date_key in SPECIAL_DATES:
        base_ndvi = base_rng.uniform(0.65, 0.95)
        base_ndbi = base_rng.uniform(0.4, 0.85)
        base_brightness = base_rng.uniform(0.4, 0.9)
        # Special dates are hotter and more humid (e.g., crowded festivals)
        base_temp += base_rng.uniform(2, 5) # even hotter
        base_humidity += base_rng.uniform(5, 10) # more humid
    else:
        base_ndvi = base_rng.uniform(-0.1, 0.6)
        base_ndbi = base_rng.uniform(-0.5, 0.4)
        base_brightness = base_rng.uniform(0.05, 0.35)

    # per-scene jitter
    if scene_id:
        scene_seed = (base_seed ^ (int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF)) & 0xFFFFFFFF
        scene_rng = random.Random(scene_seed)
        ndvi = base_ndvi + scene_rng.uniform(-0.05, 0.05)
        ndbi = base_ndbi + scene_rng.uniform(-0.05, 0.05)
        brightness = base_brightness + scene_rng.uniform(-0.05, 0.05)
        temp = base_temp + scene_rng.uniform(-1.0, 1.0)
        humidity = base_humidity + scene_rng.uniform(-2.0, 2.0)
    else:
        ndvi = base_ndvi + base_rng.uniform(-0.03, 0.03)
        ndbi = base_ndbi + base_rng.uniform(-0.03, 0.03)
        brightness = base_brightness + base_rng.uniform(-0.02, 0.02)
        temp = base_temp + base_rng.uniform(-1.0, 1.0)
        humidity = base_humidity + base_rng.uniform(-2.0, 2.0)

    # clamp to plausible bounds
    output = {
        "ndvi": max(-1.0, min(1.0, ndvi)),
        "ndbi": max(-1.0, min(1.0, ndbi)),
        "brightness": max(0.0, min(1.0, brightness)),
        "temperature_celsius": max(-10.0, min(50.0, temp)),
        "relative_humidity_percent": max(0.0, min(100.0, humidity)),
    }
    # Return float values, rounded for cleanliness
    for k in output:
        output[k] = float(f"{output[k]:.6f}")

    return output

# *** MODIFIED FUNCTION ***
def _normalize_items(items, token=None, compute_indices=False):
    normalized = []
    
    # This function is missing from your script, but it is called.
    # I am assuming it's a placeholder for a real implementation
    # that you would use if compute_indices=True.
    def compute_scene_indices(token, item):
        # Placeholder: This function would use the Process API
        # to calculate real values.
        print("Warning: compute_scene_indices is not implemented. Falling back to fake data.")
        return None, None, None

    for item in items:
        props = item.get("properties", {}) or {}
        assets = item.get("assets", {}) or {}

        cloud = None
        for key in ("eo:cloud_cover", "cloud_cover", "cloud_coverage", "cloudCoverage"):
            if key in props:
                cloud = props.get(key)
                break
        try:
            cloud = float(cloud) if cloud is not None else None
        except Exception:
            cloud = None

        dt_str = props.get("datetime")

        norm = {
            "id": item.get("id"),
            "datetime": dt_str,
            "cloud_cover": cloud,
            "platform": props.get("platform") or props.get("sat:constellation"),
            "bbox": item.get("bbox"),
            "geometry": item.get("geometry"),
            "assets": {k: v.get("href") for k, v in assets.items() if isinstance(v, dict) and v.get("href")},
            "ndvi_mean": None,
            "ndbi_mean": None,
            "brightness_mean": None,
            "temperature_celsius": None,       # NEW FIELD
            "relative_humidity_percent": None, # NEW FIELD
        }

        if compute_indices and token is not None:
            # This 'compute_scene_indices' function is not defined in
            # your script, so this path will likely fail or return None.
            ndvi_mean, ndbi_mean, bright_mean = compute_scene_indices(token, item)
            norm["ndvi_mean"] = ndvi_mean
            norm["ndbi_mean"] = ndbi_mean
            norm["brightness_mean"] = bright_mean

        # Replace NULL values with deterministic fake data
        # This block will run if compute_indices=False OR if it fails
        if norm["ndvi_mean"] is None or norm["temperature_celsius"] is None:
            
            fake_data = _generate_fake_indices(dt_str, scene_id=norm.get("id"))
            
            if norm["ndvi_mean"] is None:
                norm["ndvi_mean"] = fake_data["ndvi"]
            if norm["ndbi_mean"] is None:
                norm["ndbi_mean"] = fake_data["ndbi"]
            if norm["brightness_mean"] is None:
                norm["brightness_mean"] = fake_data["brightness"]
            if norm["temperature_celsius"] is None:
                norm["temperature_celsius"] = fake_data["temperature_celsius"]
            if norm["relative_humidity_percent"] is None:
                norm["relative_humidity_percent"] = fake_data["relative_humidity_percent"]

        normalized.append(norm)
    return normalized

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def download_thumbnail(item, thumb_dir):
    # Placeholder
    return None

def _iter_dates(start_str, end_str):
    start = datetime.fromisoformat(start_str.replace("Z", "+00:00")).date()
    end = datetime.fromisoformat(end_str.replace("Z", "+00:00")).date()
    curr = start
    while curr <= end:
        yield curr
        curr += timedelta(days=1)


def parse_bbox(s):
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be minLon,minLat,maxLon,maxLat")
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("bbox coordinates must be floats")

# --- MAIN PROGRAM ---

def main():
    parser = argparse.ArgumentParser(description="Collect Sentinel-2 scene metadata from Copernicus Data Space")
    parser.add_argument("--bbox", type=parse_bbox, default=",".join(map(str, DEFAULT_BBOX)),
                        help="minLon,minLat,maxLon,maxLat")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cloud-max", type=float, default=30.0, help="max cloud cover percentage")
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--download-thumbs", action="store_true", help="download thumbnails if available")
    parser.add_argument("--out", default="areas_output", help="output directory for per-area JSON files")
    parser.add_argument("--thumb-dir", default="thumbnails", help="thumbnail output directory")
    parser.add_argument("--daily", action="store_true", help="Save one JSON file per day in the date range (organized by area)")
    parser.add_argument("--sleep-between-days", type=float, default=0.15,
                        help="Seconds to sleep between daily requests to avoid rate limits")
    parser.add_argument("--compute-indices", action="store_true",
                        help="Compute NDVI/NDBI/Brightness per scene using Process API (requires imageio & numpy)")
    parser.add_argument("--areas", default=None,
                        help="Optional comma-separated subset of predefined areas to process (names matching keys in AREAS)")
    parser.add_argument("--single", action="store_true",
                        help="Write a single output file instead of per-area files")

    args = parser.parse_args()

    ensure_credentials()
    print("🔐 Getting access token...")
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)

    if args.areas:
        requested = [s.strip() for s in args.areas.split(",") if s.strip()]
        areas_to_run = {k: AREAS[k] for k in requested if k in AREAS}
    else:
        areas_to_run = AREAS

    if args.daily:
        print("📆 Running per-day collection from", args.start, "to", args.end)
        for area_name, area_bbox in areas_to_run.items():
            area_dir = os.path.join(args.out, area_name)
            os.makedirs(area_dir, exist_ok=True)
            print(f"▶ Area: {area_name} (bbox={area_bbox})")
            for day in _iter_dates(args.start, args.end):
                day_start = day.strftime("%Y-%m-%dT00:00:00Z")
                day_end = day.strftime("%Y-%m-%dT23:59:59Z")
                print(f"  ▶ {day.isoformat()} ...", end=" ", flush=True)
                try:
                    items = collect_results(
                        token, area_bbox, day_start, day_end, args.cloud_max,
                        per_page=args.per_page, max_items=args.max_items
                    )
                except SystemExit as e:
                    print("failed:", e)
                    time.sleep(args.sleep_between_days)
                    continue

                normalized = _normalize_items(items, token=token, compute_indices=args.compute_indices)
                out_path = os.path.join(area_dir, f"{day.strftime('%Y-%m-%d')}.json") # Use YYYY-MM-DD for filename
                _save_json(out_path, {"date": day.isoformat(), "area": area_name, "count": len(normalized), "items": normalized})
                print(f"saved {len(normalized)} -> {out_path}")
                time.sleep(args.sleep_between_days)
        return

    if args.single:
        print("🌍 Searching Sentinel-2 L2A data (single output)...")
        items = collect_results(
            token, args.bbox, args.start, args.end, args.cloud_max,
            per_page=args.per_page, max_items=args.max_items,
        )
        if not items:
            print("No results found.")
            return
        normalized = _normalize_items(items, token=token, compute_indices=args.compute_indices)
        out_path = args.out if args.out.endswith(".json") else os.path.join(args.out, "catalog_results.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _save_json(out_path, {"start": args.start, "end": args.end, "count": len(normalized), "items": normalized})
        print(f"✅ Saved {len(normalized)} scene metadata records to {out_path}")
        return

    out_base = args.out
    os.makedirs(out_base, exist_ok=True)
    print("🌍 Creating one JSON file per area for date range", args.start, "->", args.end)
    for area_name, area_bbox in areas_to_run.items():
        print(f"▶ Area: {area_name} (bbox={area_bbox}) ...", end=" ", flush=True)
        try:
            items = collect_results(
                token, area_bbox, args.start, args.end, args.cloud_max,
                per_page=args.per_page, max_items=args.max_items,
            )
        except SystemExit as e:
            print("failed:", e)
            continue

        normalized = _normalize_items(items, token=token, compute_indices=args.compute_indices)
        safe_name = area_name.replace(" ", "_")
        out_path = os.path.join(out_base, f"{safe_name}.json")
        _save_json(out_path, {"area": area_name, "coords": area_bbox, "start": args.start, "end": args.end, "count": len(normalized), "items": normalized})
        print(f"saved {len(normalized)} -> {out_path}")

    # (Thumbnail logic remains the same)
    if args.download_thumbs:
        # Note: 'items' will only contain results from the *last* area
        # in the loop above. This part may need adjustment if you
        # want to download thumbnails for *all* areas.
        os.makedirs(args.thumb_dir, exist_ok=True)
        downloaded = 0
        print(f"📥 Downloading thumbnails for last area: {area_name}...")
        # for item in items: # This 'items' is from the last loop
        #     path = download_thumbnail(item, args.thumb_dir)
        #     if path:
        #         downloaded += 1
        print(f"📥 (Thumbnail download logic is a placeholder) Downloaded {downloaded} thumbnails to {args.thumb_dir}")

if __name__ == "__main__":
    main()