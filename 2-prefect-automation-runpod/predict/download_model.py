import os
import shutil
import tempfile

import wandb

from utils.config import load_config


def download_model(config: dict = None):
    """Download trained model from W&B artifacts and save to predict/model/model.pkl."""
    if config is None:
        config = load_config("wandb")

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    api = wandb.Api()
    artifact = api.artifact(f"{config['entity']}/{config['project']}/{config['model_artifact_name']}:latest")
    artifact_dir = artifact.download(root=tempfile.gettempdir())

    output_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(
        os.path.join(artifact_dir, "model.pkl"),
        os.path.join(output_dir, "model.pkl"),
    )
    print(f"Model saved to {os.path.join(output_dir, 'model.pkl')}")


if __name__ == "__main__":
    download_model()
