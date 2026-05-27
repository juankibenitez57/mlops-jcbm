import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "/opt/ml/processing/input/titanic_raw.csv"
TRAIN_PATH = "/opt/ml/processing/output/train"
VALIDATION_PATH = "/opt/ml/processing/output/validation"

DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]
FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Fare", "Embarked", "FamilySize", "IsAlone", "FarePerPerson",
]
TARGET = "Survived"


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    df = df.dropna(subset=FEATURES + [TARGET])
    return df


if __name__ == "__main__":
    print(f"Reading raw data from {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Raw data shape: {df.shape}")

    df = clean_and_engineer(df)
    print(f"Processed data shape: {df.shape}")

    cols = [TARGET] + FEATURES
    df = df[cols]

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    print(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows")

    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(VALIDATION_PATH, exist_ok=True)

    train_df.to_csv(os.path.join(TRAIN_PATH, "train.csv"), index=False, header=False)
    val_df.to_csv(os.path.join(VALIDATION_PATH, "validation.csv"), index=False, header=False)

    print("Processing job completed successfully.")
