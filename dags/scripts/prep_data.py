# scripts/prep_data.py
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"


# ---------- helpers ----------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая очистка без инженерии признаков."""
    df = df.copy()

    # drop явно лишнее / с кучей пропусков
    for col in ["Cabin", "Ticket", "Name"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # PassengerId лучше убрать из признаков (оставим только в test, если нужно)
    if "PassengerId" in df.columns and "Survived" in df.columns:
        df = df.drop(columns=["PassengerId"])

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Простая инженерия признаков."""
    df = df.copy()

    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Грубые бины по возрасту/тарифа (работаем с числовыми значениями)
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


def one_hot_like_train(df_train: pd.DataFrame, df_test: pd.DataFrame, cat_cols):
    """One-hot из train; test выравниваем по его колонкам."""
    train_dum = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
    test_dum = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
    test_dum = test_dum.reindex(columns=train_dum.columns, fill_value=0)
    return train_dum, test_dum


# ---------- load ----------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# сохраним PassengerId из test на всякий случай
test_ids = test["PassengerId"] if "PassengerId" in test.columns else None

# ---------- imputations, одинаково для train/test ----------
# Age: median по train
if "Age" in train.columns:
    age_imp = SimpleImputer(strategy="median")
    train["Age"] = age_imp.fit_transform(train[["Age"]]).ravel()
    if "Age" in test.columns:
        test["Age"] = age_imp.transform(test[["Age"]]).ravel()

# Embarked: most frequent по train
if "Embarked" in train.columns:
    emb_imp = SimpleImputer(strategy="most_frequent")
    train["Embarked"] = emb_imp.fit_transform(train[["Embarked"]]).ravel()
    if "Embarked" in test.columns:
        test["Embarked"] = emb_imp.transform(test[["Embarked"]]).ravel()

# Fare (на тесте бывают пропуски)
if "Fare" in train.columns:
    fare_imp = SimpleImputer(strategy="median")
    train["Fare"] = fare_imp.fit_transform(train[["Fare"]]).ravel()
    if "Fare" in test.columns:
        test["Fare"] = fare_imp.transform(test[["Fare"]]).ravel()

# ---------- SIMPLE ----------
train_simple = basic_clean(train)
test_simple = basic_clean(test)

# Категориальные для one-hot в simple
simple_cat = [c for c in ["Sex", "Embarked"] if c in train_simple.columns]

# Разделим таргет, чтобы при get_dummies он не потерялся
y_train = train_simple["Survived"] if "Survived" in train_simple.columns else None
X_train_simple = (
    train_simple.drop(columns=["Survived"])
    if "Survived" in train_simple.columns
    else train_simple
)
X_test_simple = test_simple

Xtr_s, Xte_s = one_hot_like_train(X_train_simple, X_test_simple, cat_cols=simple_cat)

# вернём таргет в train
if y_train is not None:
    train_simple_final = pd.concat(
        [y_train.reset_index(drop=True), Xtr_s.reset_index(drop=True)], axis=1
    )
else:
    train_simple_final = Xtr_s

test_simple_final = Xte_s
if test_ids is not None and "PassengerId" not in test_simple_final.columns:
    test_simple_final.insert(0, "PassengerId", test_ids.values)

# ---------- FEATURE-ENGINEERED ----------
train_fe = add_features(train_simple)
test_fe = add_features(test_simple)

fe_cat = [
    c for c in ["Sex", "Embarked", "AgeGroup", "FareGroup"] if c in train_fe.columns
]

y_train = train_fe["Survived"] if "Survived" in train_fe.columns else None
X_train_fe = (
    train_fe.drop(columns=["Survived"]) if "Survived" in train_fe.columns else train_fe
)
X_test_fe = test_fe

Xtr_fe, Xte_fe = one_hot_like_train(X_train_fe, X_test_fe, cat_cols=fe_cat)

if y_train is not None:
    train_fe_final = pd.concat(
        [y_train.reset_index(drop=True), Xtr_fe.reset_index(drop=True)], axis=1
    )
else:
    train_fe_final = Xtr_fe

test_fe_final = Xte_fe
if test_ids is not None and "PassengerId" not in test_fe_final.columns:
    test_fe_final.insert(0, "PassengerId", test_ids.values)

# ---------- save ----------
train_simple_path = OUT_DIR / "train_simple.csv"
test_simple_path = OUT_DIR / "test_simple.csv"
train_fe_path = OUT_DIR / "train_fe.csv"
test_fe_path = OUT_DIR / "test_fe.csv"

train_simple_final.to_csv(train_simple_path, index=False)
test_simple_final.to_csv(test_simple_path, index=False)
train_fe_final.to_csv(train_fe_path, index=False)
test_fe_final.to_csv(test_fe_path, index=False)

lines = [
    "Saved:",
    f"- {train_simple_path}",
    f"- {test_simple_path}",
    f"- {train_fe_path}",
    f"- {test_fe_path}",
]
print("\n".join(lines))
