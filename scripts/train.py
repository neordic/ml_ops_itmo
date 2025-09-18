# scripts/train.py
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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


def train_and_save(model, X, y, out_path: Path):
    model.fit(X, y)
    joblib.dump(model, out_path)
    print(f"Saved model -> {out_path}")


def main():
    # Загрузим оба датасета
    X_s, y_s = load_Xy(TRAIN_SIMPLE)
    X_f, y_f = load_Xy(TRAIN_FE)

    # Определим три модели (минимальные параметры, без кросс-валидации)
    models = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=None, random_state=42),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42),
    }

    # simple
    for name, mdl in models.items():
        out = MODELS_DIR / f"{name}_simple.pkl"
        train_and_save(mdl, X_s, y_s, out)

    # fe
    for name, mdl in models.items():
        out = MODELS_DIR / f"{name}_fe.pkl"
        train_and_save(mdl, X_f, y_f, out)


if __name__ == "__main__":
    main()
