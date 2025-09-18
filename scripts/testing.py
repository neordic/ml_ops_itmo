# scripts/testing.py
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "simple": {
        "train": PROCESSED_DIR / "train_simple.csv",
        "test": PROCESSED_DIR / "test_simple.csv",
        "models": {
            "logreg": MODELS_DIR / "logreg_simple.pkl",
            "rf": MODELS_DIR / "rf_simple.pkl",
            "mlp": MODELS_DIR / "mlp_simple.pkl",
        },
    },
    "fe": {
        "train": PROCESSED_DIR / "train_fe.csv",
        "test": PROCESSED_DIR / "test_fe.csv",
        "models": {
            "logreg": MODELS_DIR / "logreg_fe.pkl",
            "rf": MODELS_DIR / "rf_fe.pkl",
            "mlp": MODELS_DIR / "mlp_fe.pkl",
        },
    },
}


def maybe_dvc_pull():
    """Если каких-то моделей/данных не хватает — попробуем dvc pull (мягко)."""
    needed = [
        *[p for v in DATASETS.values() for p in [v["train"], v["test"]]],
        *[p for v in DATASETS.values() for p in v["models"].values()],
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if not missing:
        return
    try:
        print("Some artifacts missing. Trying `dvc pull` …")
        subprocess.run(["dvc", "pull"], check=True)
    except Exception as e:
        print(
            f"Warning: couldn't run `dvc pull`: {e}. Proceeding with what's available.",
            file=sys.stderr,
        )


def load_xy(df: pd.DataFrame):
    """Разделить X/y. Уберём Survived и PassengerId из признаков, если они есть."""
    if "Survived" not in df.columns:
        raise ValueError("Нет столбца 'Survived' для вычисления метрик.")
    y = df["Survived"].copy()
    drop_cols = [c for c in ["Survived", "PassengerId"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    return X, y


def evaluate_models_on_df(
    models_map: dict, df: pd.DataFrame, dataset_name: str
) -> list:
    """Вернёт список строк с метриками для данного DataFrame (где есть y)."""
    X, y = load_xy(df)
    rows = []
    for model_name, model_path in models_map.items():
        if not model_path.exists():
            print(f"Warning: model file not found: {model_path}", file=sys.stderr)
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
            }
        )
    return rows


def main():
    maybe_dvc_pull()
    all_rows = []

    for name, cfg in DATASETS.items():
        train_path = cfg["train"]
        test_path = cfg["test"]
        models_map = cfg["models"]

        use_df = None
        if test_path.exists():
            df_test = pd.read_csv(test_path)
            if "Survived" in df_test.columns:
                use_df = df_test

        if use_df is None:
            # holdout 20% из train
            df_train = pd.read_csv(train_path)
            X, y = load_xy(df_train)
            _, X_te, _, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            use_df = pd.concat(
                [y_te.reset_index(drop=True), X_te.reset_index(drop=True)],
                axis=1,
            )

        rows = evaluate_models_on_df(models_map, use_df, dataset=name)
        all_rows.extend(rows)

    metrics_df = pd.DataFrame(all_rows).sort_values(
        ["dataset", "accuracy", "f1"], ascending=[True, False, False]
    )
    out_csv = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(out_csv, index=False)

    # summary как в п.2
    best = metrics_df.sort_values(["accuracy", "f1"], ascending=False).head(1)
    if not best.empty:
        line = best.iloc[0]
        m, ds = line["model"], line["dataset"]
        acc, f1 = line["accuracy"], line["f1"]
        summary = f"Лучший результат: '{m}' на '{ds}' (acc={acc}, f1={f1})."
    else:
        summary = "Не удалось посчитать метрики."

    print("\n=== Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\n" + summary)
    print(f"\nSaved metrics -> {out_csv}")


if __name__ == "__main__":
    main()
