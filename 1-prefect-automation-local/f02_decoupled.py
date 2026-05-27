from dotenv import load_dotenv
load_dotenv()

from prefect import flow, task
from prefect.logging import get_run_logger

import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from utils.config import load_config


# ── Shared tasks ─────────────────────────────────────────────────────────────

@task(name="Download Titanic raw dataset from W&B")
def load_dataset(run: wandb.sdk.wandb_run.Run, config: dict) -> pd.DataFrame:
    logger = get_run_logger()
    artifact_ref = f"{config['artifact_name']}:{config['artifact_version']}"
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    df = pd.read_csv(os.path.join(artifact_dir, config["dataset_filename"]))
    logger.info(f"Downloaded artifact '{artifact_ref}' — {len(df)} rows")
    return df


@task(name="Clean and engineer features")
def process_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger = get_run_logger()

    df = df.drop(columns=config["drop_columns"], errors="ignore")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    df = df.dropna(subset=config["features"] + [config["target"]])
    logger.info(f"Processed dataset: {df.shape[1]} columns, {len(df)} rows")
    return df


@task(name="Persist processed dataset to W&B as artifact")
def persist_processed_dataset(
    df: pd.DataFrame, run: wandb.sdk.wandb_run.Run, config: dict
) -> None:
    logger = get_run_logger()
    dataset_path = Path(tempfile.gettempdir()) / config["processed_filename"]
    df.to_csv(dataset_path, index=False)
    artifact = wandb.Artifact(
        name=config["processed_artifact_name"],
        type=config["processed_artifact_type"],
    )
    artifact.add_file(str(dataset_path))
    run.log_artifact(artifact)
    logger.info(f"Persisted processed dataset as '{config['processed_artifact_name']}'")


@task(name="Load processed dataset from W&B artifact")
def load_processed_dataset(run: wandb.sdk.wandb_run.Run, config: dict) -> pd.DataFrame:
    logger = get_run_logger()
    artifact_ref = f"{config['processed_artifact_name']}:{config['processed_artifact_version']}"
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    df = pd.read_csv(os.path.join(artifact_dir, config["processed_filename"]))
    logger.info(f"Loaded processed artifact '{artifact_ref}' — {len(df)} rows")
    return df


@task(name="Split dataset into train/test")
def split_dataset(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger = get_run_logger()
    X = df[config["features"]]
    y = df[config["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    logger.info(f"Split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


@task(name="Train RandomForest model")
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
) -> RandomForestClassifier:
    logger = get_run_logger()
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"],
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    logger.info(
        f"Trained RandomForest — n_estimators={config['n_estimators']}, "
        f"max_depth={config['max_depth']}"
    )
    return model


@task(name="Evaluate model and log metrics to W&B")
def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run: wandb.sdk.wandb_run.Run,
) -> dict:
    logger = get_run_logger()
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    run.log(metrics)
    logger.info(f"Metrics: {metrics}")
    return metrics


@task(name="Register serialized model in W&B")
def register_model(
    model: RandomForestClassifier,
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> None:
    logger = get_run_logger()
    model_path = Path(tempfile.gettempdir()) / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    artifact = wandb.Artifact(
        name=config["model_artifact_name"],
        type=config["model_artifact_type"],
        metadata={
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "random_state": config["random_state"],
            "features": config["features"],
            "target": config["target"],
        },
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    logger.info(f"Registered model artifact '{config['model_artifact_name']}'")


# ── Flow 1: Data processing (produces artifact) ───────────────────────────────

@flow(name="F02a — Decoupled: Process Titanic data")
def process_data_pipeline() -> None:
    config = load_config("f02_config")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=config["project"],
        job_type=config["process_job_type"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        wandb.save(os.path.relpath(__file__))
        df = load_dataset(run, config)
        df = process_dataset(df, config)
        persist_processed_dataset(df, run, config)


# ── Flow 2: Model training (consumes artifact) ────────────────────────────────

@flow(name="F02b — Decoupled: Train Titanic model")
def train_model_pipeline() -> None:
    config = load_config("f02_config")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=config["project"],
        job_type=config["train_job_type"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        wandb.save(os.path.relpath(__file__))
        df = load_processed_dataset(run, config)
        X_train, X_test, y_train, y_test = split_dataset(df, config)
        model = train_model(X_train, y_train, config)
        evaluate_model(model, X_test, y_test, run)
        register_model(model, run, config)


if __name__ == "__main__":
    process_data_pipeline()
    train_model_pipeline()
