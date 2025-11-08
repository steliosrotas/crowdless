"""
Compute a crowdness_score per area from JSON files in ./areas_output.

Training:
 - Read features (ndvi_mean, ndbi_mean, brightness_mean, datetime_epoch, bbox_center_x, bbox_center_y)
 - Fill missing values with column means
 - Try StandardScaler -> PCA(n_components=1) on non-constant features
 - If PCA works, map first component to [0,100] via min-max and save scaler+pca+minmax (pickled)
 - If PCA not possible, compute per-feature min-max and weights proportional to feature std, use weighted average -> [0,100]

Prediction:
 - Prefer pickled model (safe to load only your own files)
 - If PCA-based model, apply same StandardScaler + PCA + minmax mapping
 - Else use weighted-minmax approach
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

AREAS_DIR = Path("./areas_output")
OUTPUT_FILE = AREAS_DIR / "crowdness_scores.json"
MODEL_FILE = AREAS_DIR / "crowdness_model.json"
PICKLE_MODEL_FILE = AREAS_DIR / "crowdness_model.pkl"

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


def read_area_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None and (not (isinstance(v, float) and math.isnan(v)))]
    return float(pd.Series(vals).mean()) if vals else float("nan")


def safe_max_datetime_iso(iso_strs: List[Optional[str]]) -> float:
    clean = [s for s in iso_strs if s]
    if not clean:
        return float("nan")
    try:
        ts = pd.to_datetime(clean, utc=True)
        mx = ts.max()
        return float(mx.timestamp())
    except Exception:
        return float("nan")


def bbox_center_from_coords(coords: Optional[List[float]]) -> Tuple[float, float]:
    if not coords or len(coords) < 4:
        return float("nan"), float("nan")
    try:
        minx, miny, maxx, maxy = map(float, coords[:4])
        return (minx + maxx) / 2.0, (miny + maxy) / 2.0
    except Exception:
        return float("nan"), float("nan")


def aggregate_features(area_json: Dict[str, Any]) -> Dict[str, float]:
    items = area_json.get("items", [])
    ndvi = [it.get("ndvi_mean") for it in items]
    ndbi = [it.get("ndbi_mean") for it in items]
    brightness = [it.get("brightness_mean") for it in items]

    datetimes = [it.get("datetime") for it in items]
    datetime_epoch = safe_max_datetime_iso(datetimes)

    if area_json.get("coords"):
        center_x, center_y = bbox_center_from_coords(area_json.get("coords"))
    else:
        first_bbox = None
        for it in items:
            b = it.get("bbox")
            if b:
                first_bbox = b
                break
        center_x, center_y = bbox_center_from_coords(first_bbox)

    return {
        "ndvi_mean": safe_mean(ndvi),
        "ndbi_mean": safe_mean(ndbi),
        "brightness_mean": safe_mean(brightness),
        "datetime_epoch": float(datetime_epoch),
        "bbox_center_x": float(center_x),
        "bbox_center_y": float(center_y),
    }


def build_score_model(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Try PCA-based scorer first (StandardScaler -> PCA(1) -> percentile-based mapping).
    If PCA can't be used, use weighted min-max based on percentiles.
    Avoid degenerate ranges by expanding with a small epsilon.
    """
    Xf = X.astype(float).copy()
    Xf = Xf.fillna(Xf.mean())

    # identify varying features
    col_std = Xf.std(axis=0, ddof=0)
    varying = [c for c, s in col_std.items() if not math.isclose(float(s), 0.0)]
    model_meta: Dict[str, Any] = {"feature_order": list(Xf.columns)}

    # percentiles to use for robust clipping
    p_lo, p_hi = 1.0, 99.0
    quant_lo = Xf.quantile(q=p_lo / 100.0)
    quant_hi = Xf.quantile(q=p_hi / 100.0)

    if len(varying) >= 1:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xf)
        var_idx = [Xf.columns.get_loc(c) for c in varying]
        Xs_var = Xs[:, var_idx]
        pca = PCA(n_components=1)
        comp = pca.fit_transform(Xs_var).ravel()

        # If PCA explains negligible variance, or training set is tiny, use weighted fallback.
        # This prevents degenerate comp ranges that map every prediction to 50%.
        try:
            variance0 = float(pca.explained_variance_[0]) if getattr(pca, "explained_variance_", None) is not None else 0.0
        except Exception:
            variance0 = 0.0
        if variance0 < 1e-8 or len(Xf) < 4:
            return _build_weighted_model(Xf, quant_lo, quant_hi, p_lo, p_hi)

        
        # use percentiles of component for robust mapping
        comp_p1 = float(np.nanpercentile(comp, p_lo))
        comp_p99 = float(np.nanpercentile(comp, p_hi))

        # if percentiles collapse, expand them with component std or tiny eps
        comp_std = float(np.nanstd(comp))
        if math.isclose(comp_p1, comp_p99):
            eps = comp_std if comp_std > 0 else 1e-6
            comp_p1 -= eps
            comp_p99 += eps

        # also save full min/max (training min/max) so prediction can fall back to them
        comp_min = float(np.nanmin(comp)) if comp.size > 0 else comp_p1
        comp_max = float(np.nanmax(comp)) if comp.size > 0 else comp_p99
        if math.isclose(comp_min, comp_max):
            # expand tiny range so min/max mapping won't degenerate
            eps2 = max(abs(comp_min) * 1e-3, 1e-6)
            comp_min -= eps2
            comp_max += eps2

        model_meta.update({
            "kind": "pca",
            "varying": varying,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "pca_components": pca.components_.tolist(),
            "pca_mean": pca.mean_.tolist(),
            "comp_p1": comp_p1,
            "comp_p99": comp_p99,
            "comp_min": comp_min,
            "comp_max": comp_max,
            "percentile_lo": p_lo,
            "percentile_hi": p_hi,
            "quantile_lo": quant_lo.to_dict(),
            "quantile_hi": quant_hi.to_dict(),
        })
        model_obj = {"meta": model_meta, "scaler": scaler, "pca": pca, "var_idx": var_idx}
        return {"meta": model_meta, "obj": model_obj}
    else:
        return _build_weighted_model(Xf, quant_lo, quant_hi, p_lo, p_hi)


def _build_weighted_model(Xf: pd.DataFrame, quant_lo=None, quant_hi=None, p_lo=1.0, p_hi=99.0) -> Dict[str, Any]:
    # compute percentiles for robust mins/maxs
    if quant_lo is None or quant_hi is None:
        quant_lo = Xf.quantile(q=p_lo / 100.0)
        quant_hi = Xf.quantile(q=p_hi / 100.0)
    mins = Xf.min().astype(float).to_dict()
    maxs = Xf.max().astype(float).to_dict()
    # use percentile clippers for robust mapping
    clip_mins = quant_lo.astype(float).to_dict()
    clip_maxs = quant_hi.astype(float).to_dict()

    # ensure no zero-width ranges: expand tiny ranges
    for c in clip_mins.keys():
        mn = float(clip_mins[c])
        mx = float(clip_maxs[c])
        if math.isclose(mx, mn):
            eps = max(abs(mn) * 1e-3, 1e-6)
            clip_maxs[c] = mx + eps

    stds = Xf.std(axis=0, ddof=0).astype(float)
    total_std = stds.sum()
    if math.isclose(float(total_std), 0.0):
        weights = {c: 1.0 / len(stds) for c in stds.index}
    else:
        weights = {c: float(stds[c] / total_std) for c in stds.index}
    meta = {
        "kind": "weighted",
        "mins": mins,
        "maxs": maxs,
        "clip_mins": clip_mins,
        "clip_maxs": clip_maxs,
        "weights": weights,
        "feature_order": list(Xf.columns),
        "percentile_lo": p_lo,
        "percentile_hi": p_hi,
    }
    return {"meta": meta, "obj": meta}


def save_model(model_meta: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(model_meta, fh, ensure_ascii=False, indent=2)


def save_model_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def predict_from_model(model_obj: Any, input_values: Dict[str, float]) -> float:
    """
    model_obj is pickled object or JSON meta. For weighted fallback we use clip_mins/clip_maxs.
    """
    if isinstance(model_obj, dict) and "meta" in model_obj and "obj" in model_obj:
        meta = model_obj["meta"]
        obj = model_obj["obj"]
    else:
        meta = model_obj
        obj = None

    kind = meta.get("kind")
    feature_order = meta.get("feature_order", [])
    x = np.zeros(len(feature_order), dtype=float)
    for i, f in enumerate(feature_order):
        v = input_values.get(f)
        x[i] = float(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else np.nan

    if kind == "pca":
        # prefer pickled objects
        if obj is not None and isinstance(obj, dict) and "scaler" in obj and "pca" in obj:
            scaler: StandardScaler = obj["scaler"]
            pca: PCA = obj["pca"]
            var_idx: List[int] = obj["var_idx"]
            feat_means = np.array(scaler.mean_)
            mask = np.isnan(x)
            x[mask] = feat_means[mask]
            Xs = (x - scaler.mean_) / scaler.scale_
            Xs_var = Xs[var_idx]
            comp = pca.transform(Xs_var.reshape(1, -1)).ravel()[0]
        else:
            # JSON-only path
            scaler_mean = np.array(meta["scaler_mean"])
            scaler_scale = np.array(meta["scaler_scale"])
            varying = meta["varying"]
            var_idx = [feature_order.index(c) for c in varying]
            mask = np.isnan(x)
            x[mask] = scaler_mean[mask]
            Xs = (x - scaler_mean) / scaler_scale
            pca_components = np.array(meta["pca_components"])
            pca_mean = np.array(meta["pca_mean"])
            Xs_var = Xs[var_idx]
            comp = float(np.dot(Xs_var - pca_mean, pca_components[0, :]))

        comp_p1 = meta.get("comp_p1")
        comp_p99 = meta.get("comp_p99")
        # robust fallback: if percentiles degenerate, use training min/max if available
        if comp_p1 is None or comp_p99 is None or math.isclose(float(comp_p1), float(comp_p99)):
            comp_min = meta.get("comp_min")
            comp_max = meta.get("comp_max")
            if comp_min is not None and comp_max is not None and not math.isclose(float(comp_min), float(comp_max)):
                # use min-max mapping
                scaled = (comp - float(comp_min)) / (float(comp_max) - float(comp_min))
            else:
                # as ultimate fallback, expand tiny epsilon around component value
                eps = float(np.nanstd([comp])) if float(np.nanstd([comp])) > 0 else 1e-6
                comp_p1 = float(comp) - eps
                comp_p99 = float(comp) + eps
                scaled = (comp - comp_p1) / (comp_p99 - comp_p1)
        else:
            comp_p1 = float(comp_p1)
            comp_p99 = float(comp_p99)
            comp_clip = max(comp_p1, min(comp, comp_p99))
            if math.isclose(comp_p1, comp_p99):
                scaled = 0.5
            else:
                scaled = (comp_clip - comp_p1) / (comp_p99 - comp_p1)

        scaled = float(max(0.0, min(1.0, scaled)))
        return float(scaled * 100.0)
    else:
        # weighted fallback uses clip_mins/clip_maxs
        clip_mins = meta.get("clip_mins", meta.get("mins"))
        clip_maxs = meta.get("clip_maxs", meta.get("maxs"))
        weights = meta.get("weights", {})
        scaled_vals = []
        for i, f in enumerate(feature_order):
            v = x[i]
            mn = float(clip_mins.get(f, 0.0))
            mx = float(clip_maxs.get(f, 0.0))
            if np.isnan(v):
                sv = 0.5
            else:
                # robust clipping to percentile range
                v_clip = max(mn, min(v, mx))
                if math.isclose(mx, mn):
                    sv = 0.5
                else:
                    sv = (v_clip - mn) / (mx - mn)
                    sv = max(0.0, min(1.0, sv))
            scaled_vals.append(sv * weights.get(f, 0.0))
        total = sum(weights.values()) if weights else 1.0
        mean_weighted = sum(scaled_vals) / total
        return float(mean_weighted * 100.0)


def train_and_save():
    AREA_FILES = sorted(AREAS_DIR.glob("*.json"))
    # avoid reading generated outputs / model files that live in the same folder
    skip_names = {MODEL_FILE.name, OUTPUT_FILE.name}
    rows = []
    file_map = {}
    for f in AREA_FILES:
        if f.name in skip_names:
            continue
        try:
            area_json = read_area_file(f)
        except Exception:
            continue
        # require a non-empty items list (only real area files should have this)
        items = area_json.get("items")
        if not isinstance(items, list) or len(items) == 0:
            continue
        area_name = area_json.get("area") or f.stem
        feats = aggregate_features(area_json)
        rows.append({"area": area_name, **feats})
        file_map[area_name] = {"path": f, "json": area_json}

    if not rows:
        print("No area JSON files with items found in ./areas_output")
        return

    df = pd.DataFrame(rows).set_index("area")
    df = df.fillna(df.mean())

    feature_cols = [
        "ndvi_mean",
        "ndbi_mean",
        "brightness_mean",
        "datetime_epoch",
        "bbox_center_x",
        "bbox_center_y",
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = float("nan")
    df = df[feature_cols]

    model_result = build_score_model(df)
    meta = model_result["meta"]
    obj = model_result["obj"]

    # create training outputs for human inspection
    if meta.get("kind") == "pca" and isinstance(obj, dict):
        # compute scores on training set
        scaler: StandardScaler = obj["scaler"]
        pca: PCA = obj["pca"]
        var_idx: List[int] = obj["var_idx"]
        Xs = scaler.transform(df.values)
        comp = pca.transform(Xs[:, var_idx]).ravel()
        comp_min, comp_max = float(np.nanmin(comp)), float(np.nanmax(comp))
        mm = MinMaxScaler(feature_range=(0, 100))
        if not math.isclose(comp_min, comp_max):
            scores = ((comp - comp_min) / (comp_max - comp_min)) * 100.0
        else:
            scores = np.full_like(comp, 50.0)
    else:
        # weighted meta
        mins = meta["mins"]
        maxs = meta["maxs"]
        weights = meta["weights"]
        scores = []
        for _, row in df.iterrows():
            s = 0.0
            for f in feature_cols:
                mn = float(mins.get(f, 0.0))
                mx = float(maxs.get(f, 0.0))
                v = float(row[f])
                if math.isclose(mx, mn):
                    sv = 0.5
                else:
                    sv = (v - mn) / (mx - mn)
                    sv = max(0.0, min(1.0, sv))
                s += sv * weights.get(f, 0.0)
            total = sum(weights.values()) if weights else 1.0
            scores.append((s / total) * 100.0)
        scores = np.array(scores)

    # write area outputs and save models
    output = []
    for i, area in enumerate(df.index):
        entry = {
            "area": area,
            "coords": AREA_COORDS.get(area),
            "ndvi_mean": float(df.loc[area, "ndvi_mean"]),
            "ndbi_mean": float(df.loc[area, "ndbi_mean"]),
            "brightness_mean": float(df.loc[area, "brightness_mean"]),
            "datetime_epoch": float(df.loc[area, "datetime_epoch"]),
            "bbox_center_x": float(df.loc[area, "bbox_center_x"]),
            "bbox_center_y": float(df.loc[area, "bbox_center_y"]),
            "crowdness_score": float(scores[i]) if len(scores) > i else 0.0,
        }
        output.append(entry)
        fm = file_map.get(area)
        if fm:
            area_json = fm["json"]
            area_json["crowdness_score"] = float(entry["crowdness_score"])
            try:
                with fm["path"].open("w", encoding="utf-8") as fh:
                    json.dump(area_json, fh, ensure_ascii=False, indent=2)
            except Exception:
                pass

    try:
        with OUTPUT_FILE.open("w", encoding="utf-8") as outfh:
            json.dump({"areas": output}, outfh, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed to write output:", e)
        return

    # save meta JSON and full pickled object
    save_model(meta, MODEL_FILE)
    # save full object for exact behaviour (sklearn objects preserved)
    save_model_pickle({"meta": meta, "obj": obj}, PICKLE_MODEL_FILE)

    print("Trained and saved model to", MODEL_FILE, "and", PICKLE_MODEL_FILE)
    print("Computed crowdness_score for areas:")
    for a, s in zip(df.index, scores):
        print(f"  {a}: {float(s):.2f}")


def parse_bbox(s: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 4 comma-separated numbers: minx,miny,maxx,maxy")
    return tuple(float(p) for p in parts)  # type: ignore


def input_to_feature_dict(
    ndvi: Optional[float],
    ndbi: Optional[float],
    brightness: Optional[float],
    bbox: Optional[Tuple[float, float, float, float]],
    datetime_iso: Optional[str],
) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    feat["ndvi_mean"] = float(ndvi) if ndvi is not None else float("nan")
    feat["ndbi_mean"] = float(ndbi) if ndbi is not None else float("nan")
    feat["brightness_mean"] = float(brightness) if brightness is not None else float("nan")
    if bbox:
        cx, cy = bbox_center_from_coords(list(bbox))
        feat["bbox_center_x"] = float(cx)
        feat["bbox_center_y"] = float(cy)
    else:
        feat["bbox_center_x"] = float("nan")
        feat["bbox_center_y"] = float("nan")
    if datetime_iso:
        try:
            feat["datetime_epoch"] = float(pd.to_datetime(datetime_iso, utc=True).timestamp())
        except Exception:
            feat["datetime_epoch"] = float("nan")
    else:
        feat["datetime_epoch"] = float("nan")
    return feat


def cmd_predict(args):
    # prefer pickle if available
    if PICKLE_MODEL_FILE.exists():
        model_obj = load_model_pickle(PICKLE_MODEL_FILE)
    elif MODEL_FILE.exists():
        model_obj = load_model(MODEL_FILE)
    else:
        print("Model not found. Run: python ml.py train")
        return
    bbox = parse_bbox(args.bbox) if args.bbox else None
    feat = input_to_feature_dict(args.ndvi, args.ndbi, args.brightness, bbox, args.datetime)
    score = predict_from_model(model_obj, feat)
    print(f"crowdness_score: {score:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train or predict crowdness_score")
    sub = parser.add_subparsers(dest="cmd")
    p_train = sub.add_parser("train", help="Train model from ./areas_output JSON files and save model")
    p_pred = sub.add_parser("predict", help="Predict crowdness_score from saved model")
    p_pred.add_argument("--ndvi", type=float, required=False, help="ndvi_mean")
    p_pred.add_argument("--ndbi", type=float, required=False, help="ndbi_mean")
    p_pred.add_argument("--brightness", type=float, required=False, help="brightness_mean")
    p_pred.add_argument("--bbox", type=str, required=False, help="bbox as minx,miny,maxx,maxy")
    p_pred.add_argument("--datetime", type=str, required=False, help="ISO datetime string (UTC)")

    args = parser.parse_args()
    if args.cmd == "train" or args.cmd is None:
        train_and_save()
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()