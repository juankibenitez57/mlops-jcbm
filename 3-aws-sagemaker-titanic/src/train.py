import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Fare", "Embarked", "FamilySize", "IsAlone", "FarePerPerson",
]
TARGET = "Survived"


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args, _ = parser.parse_known_args()

    print("Loading training data")
    train_df = pd.read_csv(
        os.path.join(args.train, "train.csv"),
        header=None,
        names=[TARGET] + FEATURES,
    )

    print("Loading validation data")
    val_df = pd.read_csv(
        os.path.join(args.validation, "validation.csv"),
        header=None,
        names=[TARGET] + FEATURES,
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET]

    print(f"Training RandomForest — n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_val, y_pred, zero_division=0):.4f}")

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")
