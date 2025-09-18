# scripts/train.py
from pathlib import Path


import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import os

import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SIMPLE = PROCESSED_DIR / "train_simple.csv"
TRAIN_FE = PROCESSED_DIR / "train_fe.csv"


def load_Xy(train_path: Path):
    df = pd.read_csv(train_path)
    if "Survived" not in df.columns:
        raise ValueError(f"'Survived' not found in {train_path}")
    # на всякий случай уберём PassengerId, если вдруг оказался в train
    drop_cols = [c for c in ["Survived", "PassengerId"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["Survived"]
    return X, y


# def train_and_save(model, X, y, out_path: Path):
#     model.fit(X, y)
#     joblib.dump(model, out_path)
#     print(f"Saved model -> {out_path}")


def train_and_log(model, X, y, run_name: str, out_path: Path):
    """Обучаем модель, логируем метрики в MLflow и сохраняем в models/ для DVC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name=run_name):
        acc = cross_val_score(
            model, X, y, cv=cv, scoring=make_scorer(accuracy_score), n_jobs=-1
        ).mean()
        f1 = cross_val_score(
            model, X, y, cv=cv, scoring=make_scorer(f1_score), n_jobs=-1
        ).mean()
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

        mlflow.log_metric("cv_accuracy", float(acc))
        mlflow.log_metric("cv_f1", float(f1))
        mlflow.log_metric("cv_roc_auc", float(auc))

        model.fit(X, y)
        joblib.dump(model, out_path)
        print(f"Saved model -> {out_path}")


def main():
    # Настройка MLflow (по умолчанию пишет в ./mlruns или в значение из env)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops_itmo"))
    mlflow.sklearn.autolog(log_models=True)  # автолог
    # Загрузим оба датасета
    X_s, y_s = load_Xy(TRAIN_SIMPLE)
    X_f, y_f = load_Xy(TRAIN_FE)

    # Определим три модели (минимальные параметры, без кросс-валидации)
    models = {
        "logreg": LogisticRegression(
            max_iter=1000, solver="lbfgs", n_jobs=-1, random_state=42
        ),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42),
    }

    # simple
    for name, mdl in models.items():
        out = MODELS_DIR / f"{name}_simple.pkl"
        # train_and_save(mdl, X_s, y_s, out)
        train_and_log(mdl, X_s, y_s, f"{name}_simple", out)

    # fe
    for name, mdl in models.items():
        out = MODELS_DIR / f"{name}_fe.pkl"
        # train_and_save(mdl, X_f, y_f, out)
        train_and_log(mdl, X_f, y_f, f"{name}_fe", out)


if __name__ == "__main__":
    main()
