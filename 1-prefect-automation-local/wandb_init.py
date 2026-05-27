from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import wandb
from pathlib import Path

from utils.config import load_config


def main() -> None:
    config = load_config("wandb_init")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=config["project"],
        job_type="upload-dataset",
        dir=tempfile.gettempdir(),
    ):
        artifact = wandb.Artifact(
            name=config["artifact_name"],
            type=config["artifact_type"],
        )
        artifact.add_file(str(Path("utils/data/titanic_raw.csv")))
        wandb.log_artifact(artifact)
        print(f"Uploaded '{config['artifact_name']}' to W&B project '{config['project']}'")


if __name__ == "__main__":
    main()
