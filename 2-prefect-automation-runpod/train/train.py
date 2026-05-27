from dotenv import load_dotenv
load_dotenv()

import logging
import os
import pickle
import requests
import tempfile
from pathlib import Path
from time import sleep

import pandas as pd
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from utils.config import load_config

logger = logging.getLogger(__name__)

def load_processed_dataset(run: wandb.sdk.wandb_run.Run, config: dict) -> pd.DataFrame:
    artifact_ref = f"{config['processed_artifact_name']}:{config['processed_artifact_version']}"
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    df = pd.read_csv(os.path.join(artifact_dir, config["processed_filename"]))
    logger.info(f"Loaded processed artifact '{artifact_ref}' — {len(df)} rows")
    return df


def split_dataset(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
) -> RandomForestClassifier:
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


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run: wandb.sdk.wandb_run.Run,
) -> dict:
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


def register_model(
    model: RandomForestClassifier,
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> None:
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


def train_model_pipeline() -> None:
    config = {**load_config("wandb"), **load_config("training")}
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=config["project"],
        job_type=config["train_job_type"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        df = load_processed_dataset(run, config)
        X_train, X_test, y_train, y_test = split_dataset(df, config)
        model = train_model(X_train, y_train, config)
        evaluate_model(model, X_test, y_test, run)
        register_model(model, run, config)


def stop_pod() -> None:
    pod_id = os.getenv("RUNPOD_POD_ID")
    api_key = os.getenv("RUNPOD_MANAGE_KEY")
    if not pod_id or not api_key:
        logger.warning("RUNPOD_POD_ID or RUNPOD_MANAGE_KEY not set, skipping pod stop")
        return
    url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    logger.info(f"Pod {pod_id} stopped successfully")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model_pipeline()
    logger.info("Sleeping before pod stop to allow log inspection")
    total_sleep = 60  # To allow log inspection before Pod shutdown
    interval = 10
    for remaining in range(total_sleep, 0, -interval):
        logger.info(f"{remaining} seconds until pod stops")
        sleep(interval if remaining >= interval else remaining)
    stop_pod() #To avoid repeated execution