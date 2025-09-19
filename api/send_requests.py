# api/quick_bench.py
import numpy as np
import time
import statistics
import requests
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score


API_HOST = "http://localhost:8001"
MODEL_KEY = "logreg_fe"
TEST_DATA_PATH = "data/raw/test.csv"
TARGET_PATH = "data/raw/gender_submission.csv"  # метки по PassengerId
BATCH_SIZE = 64
OUT_CSV = "results/api_benchmark.csv"


def main():
    # Данные и метки
    df = pd.read_csv(TEST_DATA_PATH)
    targets = pd.read_csv(TARGET_PATH).set_index("PassengerId")["Survived"]

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)

    rows = df.to_dict(orient="records")

    # Запросы батчами
    url = f"{API_HOST.rstrip('/')}/predict?model={MODEL_KEY}"
    sess = requests.Session()
    timings, preds = [], []
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        t0 = time.perf_counter()
        r = sess.post(url, json={"records": chunk}, timeout=60)
        r.raise_for_status()
        timings.append(time.perf_counter() - t0)
        preds.extend(r.json()["predictions"])

    ids = df["PassengerId"].tolist()
    y_true = [int(targets.get(int(pid), -1)) for pid in ids]
    mask = [v != -1 for v in y_true]
    y = [v for v, m in zip(y_true, mask) if m]
    yhat = [p for p, m in zip(preds, mask) if m]

    acc = accuracy_score(y, yhat) if y else None
    f1 = f1_score(y, yhat) if y else None

    # Время
    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * (len(timings) - 1))] if len(timings) > 1 else p50
    rps = len(rows) / sum(timings)

    print(f"model={MODEL_KEY}  n={len(rows)}  batch={BATCH_SIZE}")
    print(f"latency_p50_s={p50:.4f}  latency_p95_s={p95:.4f}  throughput_rps={rps:.2f}")
    if acc is not None:
        print(f"accuracy={acc:.4f}  f1={f1:.4f}")

    # отчет
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": MODEL_KEY,
        "n": len(rows),
        "batch": BATCH_SIZE,
        "lat_p50_ms": round(p50 * 1000, 3),
        "lat_p95_ms": round(p95 * 1000, 3),
        "rps": round(rps, 3),
        "accuracy": None if acc is None else round(acc, 4),
        "f1": None if f1 is None else round(f1, 4),
        "source_csv": TEST_DATA_PATH,
        "labels": TARGET_PATH,
    }
    out = Path(OUT_CSV)
    (
        pd.concat([pd.read_csv(out), pd.DataFrame([row])], ignore_index=True)
        if out.exists()
        else pd.DataFrame([row])
    ).to_csv(out, index=False)
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
