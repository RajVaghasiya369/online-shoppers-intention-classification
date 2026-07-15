"""
Flask web app for Online Shoppers Intention prediction.

Loads all saved tuned models + the best model and a fitted preprocessor.
Exposes:
  - GET  /          HTML form with model selection
  - POST /predict   JSON / form API (optional `model` field)
  - GET  /health    Health / readiness check
  - GET  /models    List available models
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "src" / "component" / "models"
BEST_MODEL_DIR = MODELS_DIR / "best_models"
TUNED_MODEL_DIR = MODELS_DIR / "tuned_models"
PREPROCESSOR_DIR = MODELS_DIR / "preprocessors"

sys.path.insert(0, str(BASE_DIR / "src" / "component"))

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
REQUIRED_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

LOG_SOURCE_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "TrafficType",
    "TotalPages",
    "TotalDuration",
    "ValuePerProduct",
    "PageValuePerDuration",
    "ProductTimeRatio",
    "BounceExitOff",
]

DROP_COLS = ["Informational", "SpecialDay", "Weekend", "Revenue"]

VALID_MONTHS = {
    "Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
}
VALID_VISITOR_TYPES = {"Returning_Visitor", "New_Visitor", "Other"}
MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------------------
# models_registry: { "AdaBoost": {"estimator": ..., "path": "...", "is_best": True}, ... }
models_registry: Dict[str, Dict[str, Any]] = {}
default_model_name: Optional[str] = None
preprocessor = None
preprocessor_path: Optional[str] = None

# Timestamps look like _20260708_173351 at end of stem
_TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _parse_model_name(path: Path, prefix: str = "") -> str:
    """
    Extract clean model name from filename.
    AdaBoost_20260708_173351.joblib -> AdaBoost
    best_model_AdaBoost_20260708_173351.joblib -> AdaBoost
    """
    stem = path.stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):]
    stem = _TIMESTAMP_RE.sub("", stem)
    return stem


def _discover_model_files() -> List[Tuple[str, Path, bool]]:
    """
    Return list of (name, path, is_best).
    Prefers best_models entry when the same name appears in both dirs.
    """
    found: Dict[str, Tuple[Path, bool]] = {}

    if TUNED_MODEL_DIR.exists():
        for path in TUNED_MODEL_DIR.glob("*.joblib"):
            name = _parse_model_name(path)
            prev = found.get(name)
            if prev is None or path.stat().st_mtime > prev[0].stat().st_mtime:
                found[name] = (path, False)

    if BEST_MODEL_DIR.exists():
        for path in BEST_MODEL_DIR.glob("best_model_*.joblib"):
            name = _parse_model_name(path, prefix="best_model_")
            found[name] = (path, True)

    return [(name, path, is_best) for name, (path, is_best) in sorted(found.items())]


def load_artifacts() -> None:
    """Load preprocessor and all available models into the registry."""
    global models_registry, default_model_name, preprocessor, preprocessor_path

    prep_file = _latest_file(PREPROCESSOR_DIR, "data_preprocessor_*.joblib")
    if prep_file is None:
        raise FileNotFoundError(
            f"No preprocessor found in {PREPROCESSOR_DIR}. "
            "Run: python src/component/classification.py"
        )

    discovered = _discover_model_files()
    if not discovered:
        raise FileNotFoundError(
            f"No models found in {BEST_MODEL_DIR} or {TUNED_MODEL_DIR}. "
            "Run: python src/component/classification.py"
        )

    preprocessor = joblib.load(prep_file)
    preprocessor_path = str(prep_file)

    registry: Dict[str, Dict[str, Any]] = {}
    best_name: Optional[str] = None

    for name, path, is_best in discovered:
        estimator = joblib.load(path)
        registry[name] = {
            "estimator": estimator,
            "path": str(path),
            "is_best": is_best,
        }
        if is_best:
            best_name = name

    models_registry = registry
    default_model_name = best_name or sorted(registry.keys())[0]


def available_model_names() -> List[str]:
    """Ordered list: best first, then alphabetical."""
    names = list(models_registry.keys())
    if default_model_name and default_model_name in names:
        names.remove(default_model_name)
        return [default_model_name] + sorted(names)
    return sorted(names)


def get_model(name: Optional[str] = None):
    """Return (model_name, estimator) for the requested or default model."""
    if not models_registry:
        raise RuntimeError("Model artifacts are not loaded.")

    selected = (name or "").strip() or default_model_name
    if selected not in models_registry:
        raise ValueError(
            f"Unknown model '{selected}'. Available: {', '.join(available_model_names())}"
        )
    return selected, models_registry[selected]["estimator"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalPages"] = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    )
    df["TotalDuration"] = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )
    df["ValuePerProduct"] = df["PageValues"] / (df["ProductRelated"] + 1)
    df["PageValuePerDuration"] = df["PageValues"] / (df["TotalPages"] + 1)
    df["ProductFocusRatio"] = df["ProductRelated"] / (df["TotalPages"] + 1)
    df["ProductTimeRatio"] = (
        df["ProductRelated_Duration"] * df["ProductRelated"] + 1
    )
    df["BounceExitOff"] = df["BounceRates"] + df["ExitRates"]

    for col in LOG_SOURCE_COLS:
        if col in df.columns:
            df[col + "_log"] = np.log1p(df[col].astype(float).clip(lower=0))

    return df


def prepare_features(payload: dict) -> pd.DataFrame:
    missing = [f for f in REQUIRED_FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    row = {}
    for col in NUMERIC_FEATURES:
        try:
            row[col] = float(payload[col])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for '{col}': {payload[col]}") from exc

    month = str(payload["Month"]).strip()
    if month not in VALID_MONTHS:
        raise ValueError(
            f"Invalid Month '{month}'. Expected one of: {sorted(VALID_MONTHS)}"
        )
    row["Month"] = month

    visitor = str(payload["VisitorType"]).strip()
    if visitor not in VALID_VISITOR_TYPES:
        raise ValueError(
            f"Invalid VisitorType '{visitor}'. "
            f"Expected one of: {sorted(VALID_VISITOR_TYPES)}"
        )
    row["VisitorType"] = visitor

    weekend_raw = payload["Weekend"]
    if isinstance(weekend_raw, bool):
        row["Weekend"] = int(weekend_raw)
    elif isinstance(weekend_raw, (int, float)):
        row["Weekend"] = int(weekend_raw)
    elif str(weekend_raw).strip().lower() in {"true", "1", "yes"}:
        row["Weekend"] = 1
    elif str(weekend_raw).strip().lower() in {"false", "0", "no"}:
        row["Weekend"] = 0
    else:
        raise ValueError(f"Invalid Weekend value: {weekend_raw}")

    df = pd.DataFrame([row])
    df = feature_engineering(df)
    drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=drop)


def predict_from_payload(payload: dict) -> dict:
    if preprocessor is None:
        raise RuntimeError("Preprocessor is not loaded.")

    selected_name, estimator = get_model(payload.get("model"))
    X = prepare_features(payload)
    X_proc = preprocessor.transform(X)
    pred = int(estimator.predict(X_proc)[0])

    probability = None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_proc)[0]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])

    return {
        "prediction": pred,
        "revenue": bool(pred),
        "label": "Purchase Likely" if pred == 1 else "No Purchase",
        "probability": probability,
        "model": selected_name,
        "is_best": bool(models_registry[selected_name].get("is_best")),
    }


def _template_context(**extra):
    return {
        "default_model": default_model_name or "Not loaded",
        "available_models": available_model_names(),
        "months": MONTH_ORDER,
        "visitor_types": sorted(VALID_VISITOR_TYPES),
        "form_data": {},
        **extra,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", **_template_context())


@app.route("/models")
def list_models():
    return jsonify(
        {
            "default_model": default_model_name,
            "models": [
                {
                    "name": name,
                    "path": models_registry[name]["path"],
                    "is_best": models_registry[name]["is_best"],
                }
                for name in available_model_names()
            ],
        }
    )


@app.route("/health")
def health():
    ready = bool(models_registry) and preprocessor is not None
    return jsonify(
        {
            "status": "ok" if ready else "degraded",
            "models_loaded": len(models_registry),
            "available_models": available_model_names(),
            "default_model": default_model_name,
            "preprocessor_loaded": preprocessor is not None,
            "preprocessor_path": preprocessor_path,
        }
    ), (200 if ready else 503)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            payload = request.get_json(force=True) or {}
        else:
            payload = request.form.to_dict()

        result = predict_from_payload(payload)

        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({"success": True, **result})

        return render_template(
            "index.html",
            **_template_context(result=result, form_data=payload),
        )

    except ValueError as exc:
        message = str(exc)
        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({"success": False, "error": message}), 400
        return render_template(
            "index.html",
            **_template_context(error=message, form_data=request.form.to_dict()),
        ), 400

    except Exception as exc:
        message = f"Prediction failed: {exc}"
        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({"success": False, "error": message}), 500
        return render_template(
            "index.html",
            **_template_context(
                error=message,
                form_data=request.form.to_dict() if request.form else {},
            ),
        ), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    load_artifacts()
    return app


try:
    load_artifacts()
    print(f"Loaded {len(models_registry)} model(s). Default: {default_model_name}")
    for name in available_model_names():
        info = models_registry[name]
        tag = " [best]" if info["is_best"] else ""
        print(f"  - {name}{tag}: {info['path']}")
    print(f"  Preprocessor: {preprocessor_path}")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    print("The /predict endpoint will fail until models are trained.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
