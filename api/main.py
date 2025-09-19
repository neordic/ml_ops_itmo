from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
import os
import yaml
import subprocess
import pandas as pd
import joblib

import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQ_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of /predict by model",
    ["model"],
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0),
)
REQ_TOTAL = Counter(
    "api_requests_total", "Requests count by model/status", ["model", "status"]
)

app = FastAPI(title="Titanic API", version="1.0.0")


CFG_PATH = os.getenv("API_CONFIG", "api/config.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DEFAULT_KEY = os.getenv("MODEL_KEY", CFG.get("default_model"))
MODELS_CFG: Dict[str, Dict] = CFG["models"]
MODEL_CACHE: Dict[str, dict] = {}


def _ensure_from_dvc(
    dvc_path: str, git_rev: str, cache_dir: Path, out_name: str
) -> Path:
    """
    HEAD -> dvc pull <path> в рабочее дерево.
    <rev> -> dvc get --rev <rev> в локальный кэш сервиса.
    """
    p = Path(dvc_path)
    if git_rev in (None, "", "HEAD"):
        if not p.exists():
            try:
                subprocess.run(["dvc", "pull", dvc_path], check=True)
            except Exception:
                pass
        if not p.exists():
            raise FileNotFoundError(dvc_path)
        return p

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / f"{out_name}__{git_rev}.pkl"
    if not out_file.exists():
        subprocess.run(
            ["dvc", "get", "--rev", git_rev, ".", dvc_path, "-o", str(out_file)],
            check=True,
        )
    return out_file


def ensure_model_loaded(key: str):
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    if key not in MODELS_CFG:
        raise KeyError(f"unknown model key: {key}")
    cfg = MODELS_CFG[key]
    dvc_path = cfg["dvc_path"]
    git_rev = cfg.get("git_rev", "HEAD")
    local_path = _ensure_from_dvc(dvc_path, git_rev, Path("/app/.model_cache"), key)
    mdl = joblib.load(local_path)
    features = list(getattr(mdl, "feature_names_in_", []))
    MODEL_CACHE[key] = {"model": mdl, "features": features}
    return MODEL_CACHE[key]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Cabin", "Ticket", "Name"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    if "PassengerId" in df.columns and "Survived" in df.columns:
        df.drop(columns=["PassengerId"], inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 12, 18, 30, 50, 200],
            labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
        )
    if "Fare" in df.columns:
        df["FareGroup"] = pd.cut(
            df["Fare"], bins=[0, 10, 30, 1000], labels=["Low", "Medium", "High"]
        )
    return df


def to_model_X(df: pd.DataFrame, key: str) -> pd.DataFrame:
    cfg = MODELS_CFG[key]
    df = df.copy()
    for c in cfg.get("drop_if_present", []):
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    df = basic_clean(df)
    preset = cfg.get("preset", "simple")
    if preset == "fe":
        df = add_features(df)
        cat = [
            c for c in ["Sex", "Embarked", "AgeGroup", "FareGroup"] if c in df.columns
        ]
    else:
        cat = [c for c in ["Sex", "Embarked"] if c in df.columns]
    df_enc = pd.get_dummies(df, columns=cat, drop_first=True)
    features = ensure_model_loaded(key)["features"]
    if features:
        df_enc = df_enc.reindex(columns=features, fill_value=0)
    return df_enc


class PredictRequest(BaseModel):
    records: List[Dict]


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float] | None = None


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "default_model": DEFAULT_KEY, "available": list(MODELS_CFG)}


# @app.post("/predict", response_model=PredictResponse)
# def predict(req: PredictRequest, model: str = Query(default=DEFAULT_KEY)):
#     if model not in MODELS_CFG:
#         raise HTTPException(400, f"unknown model key: {model}")
#     try:
#         df = pd.DataFrame(req.records)
#     except Exception as e:
#         raise HTTPException(400, f"bad payload: {e}")


#     X = to_model_X(df, model)
#     obj = ensure_model_loaded(model)
#     mdl = obj["model"]
#     proba = getattr(mdl, "predict_proba", None)
#     probs = mdl.predict_proba(X)[:, 1].tolist() if proba else None
#     preds = mdl.predict(X).tolist()
#     return {"predictions": preds, "probabilities": probs}
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model: str = Query(default=DEFAULT_KEY)):
    t0 = time.perf_counter()
    try:
        if model not in MODELS_CFG:
            REQ_TOTAL.labels(model=model, status="400").inc()
            raise HTTPException(400, f"unknown model key: {model}")

        df = pd.DataFrame(req.records)
        X = to_model_X(df, model)
        obj = ensure_model_loaded(model)
        mdl = obj["model"]
        proba = getattr(mdl, "predict_proba", None)
        probs = mdl.predict_proba(X)[:, 1].tolist() if proba else None
        preds = mdl.predict(X).tolist()

        REQ_TOTAL.labels(model=model, status="200").inc()
        return {"predictions": preds, "probabilities": probs}

    except HTTPException:
        REQ_TOTAL.labels(model=model, status="400").inc()
        raise
    except Exception as e:
        REQ_TOTAL.labels(model=model, status="500").inc()
        raise HTTPException(500, str(e))
    finally:
        REQ_LATENCY.labels(model=model).observe(time.perf_counter() - t0)
