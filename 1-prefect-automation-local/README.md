# Introduction to Prefect

MLOps project that uses **Prefect** for workflow orchestration and **Weights & Biases (W&B)** for experiment tracking and model registry. The project demonstrates two approaches to building ML pipelines with the Iris dataset: a monolithic pipeline and a decoupled pipeline.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Environment Variables](#environment-variables)
- [Starting Prefect](#starting-prefect)
- [Initializing W&B with a Dataset](#initializing-wb-with-a-dataset)
- [Deploying Flows](#deploying-flows)
- [Existing Flows](#existing-flows)
- [Creating New Flows](#creating-new-flows)

## Project Structure

```
├── deploy_flows.py          # Deploys all Prefect flows
├── docker-compose.yml       # Prefect server, Postgres, Redis infrastructure
├── f01_monolith.py          # Monolith pipeline: single flow for the full ML lifecycle
├── f02_decoupled.py         # Decoupled pipeline: separate flows for data processing and training
├── wandb_init.py            # Uploads initial dataset to W&B as an artifact
├── pyproject.toml           # Project dependencies (managed with uv)
├── .env.example             # Template for environment variables
├── config/
│   ├── f01_config.json      # Configuration for the monolith pipeline
│   ├── f02_config.json      # Configuration for the decoupled pipeline
│   └── wandb_init.json      # Configuration for W&B dataset upload
└── utils/
    ├── __init__.py
    ├── config.py            # load_config() helper
    └── data/
        ├── original.csv     # Original Iris dataset
        └── changed-labels.csv  # Iris dataset with modified labels
```

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (must be running before starting Prefect)
- A **[Weights & Biases](https://wandb.ai/)** account

## Environment Setup

Install all project dependencies with `uv`:

```bash
uv sync
```

This reads `pyproject.toml` and creates a virtual environment (`.venv/`) with all required packages: Prefect, W&B, scikit-learn, pandas, etc.


## Environment Variables

Rename the example file to create your own `.env`:

Edit `.env` and fill in the actual values:

```
PREFECT_API_URL=http://127.0.0.1:4200/api
WANDB_API_KEY=wandb_v1_YOUR_ACTUAL_KEY
```

| Variable | Description |
| --- | --- |
| `PREFECT_API_URL` | URL of the Prefect API server. Keep the default when running locally with Docker Compose. |
| `WANDB_API_KEY` | Your W&B API key. Find it at [wandb.ai/authorize](https://wandb.ai/authorize). |

> **Note:** Never commit the `.env` file. It is already listed in `.gitignore`.

## Starting Prefect

Prefect runs on Docker via the provided `docker-compose.yml`, which spins all background **services**, and a **worker**.

1. (If needed) **Make sure Docker Desktop is running.**

2. **Start the infrastructure:**

   ```bash
   docker compose up -d
   ```

3. **Verify all containers are healthy:**

   ```bash
   docker compose ps
   ```

   You should see `postgres`, `redis`, `prefect-server`, `prefect-services`, and `prefect-worker` all running.

4. **Open the Prefect UI** at [http://localhost:4200](http://localhost:4200).

To stop the infrastructure:

```bash
docker compose down
```

To stop and remove all data (volumes):

```bash
docker compose down -v
```

## Initializing W&B with a Dataset

Before running any pipeline, you need to upload the Iris dataset to W&B as an artifact. This is a one-time setup step.

Run the initialization script:

```bash
uv run python wandb_init.py
```

This uploads the file `original.csv` from `utils/data/` to your W&B project (`iris-quickstart` by default) as an artifact named `raw_data`. You can change these values in `config/wandb_init.json`:

```json
{
    "project": "iris-quickstart",
    "artifact_name": "raw_data",
    "artifact_type": "dataset"
}
```

## Deploying Flows

With Prefect infrastructure running, deploy all flows:

```bash
uv run python deploy_flows.py
```

This registers the following deployments with the Prefect server:

| Deployment | Source |
| --- | --- |
| Monolith - Full pipeline | `f01_monolith.py` |
| Decoupled - Process data | `f02_decoupled.py` |
| Decoupled - Train model | `f02_decoupled.py` |

Once deployed, you can trigger runs from the [Prefect UI](http://localhost:4200) or via the Prefect CLI.

## Existing Flows

### f01 — Monolith Pipeline (`f01_monolith.py`)

A single flow (`full_pipeline`) that executes the entire ML lifecycle in sequence:

1. Download the raw Iris dataset from W&B
2. Process the data (drop columns, create area features)
3. Split into train/test sets
4. Train a KNN classifier
5. Evaluate the model (accuracy, precision, recall, F1)
6. Register the trained model as a W&B artifact

Configuration: `config/f01_config.json`

### f02 — Decoupled Pipeline (`f02_decoupled.py`)

Two independent flows that separate data processing from model training:

- **`process_data_pipeline`**: Downloads raw data, processes it, and persists the result as a new W&B artifact (`processed_data`).
- **`train_model_pipeline`**: Loads the processed artifact, splits it, trains a KNN model, evaluates it, and registers the model.

This pattern allows you to run data processing and training independently, and to retrain models without reprocessing data.

Configuration: `config/f02_config.json`

## Creating New Flows

Follow these steps to add a new pipeline to the project:

### 1. Create the configuration file

Add a new JSON file in `config/` (e.g., `config/f03_config.json`) with all tunable parameters. Never hardcode configuration values in Python files.

```json
{
    "project": "iris-quickstart",
    "job_type": "my-job",
    "artifact_name": "raw_data",
    "artifact_version": "latest"
}
```

### 2. Create the flow file

Create a new Python file at the project root following the naming convention `fXX_name.py` (e.g., `f03_my_pipeline.py`).

```python
from dotenv import load_dotenv
load_dotenv()

from prefect import flow, task
from prefect.logging import get_run_logger

import os
import tempfile

import wandb

from utils.config import load_config


@task(name="My task")
def my_task(config: dict) -> None:
    logger = get_run_logger()
    logger.info("Running my task")


@flow(name="My Pipeline")
def my_pipeline() -> None:
    config = load_config("f03_config")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        project=config["project"],
        job_type=config["job_type"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        my_task(config)
```

Key rules:
- Use `@flow` for orchestrators and `@task` for logic.
- Use `get_run_logger()` for logging inside tasks and flows.
- Load configuration with `load_config()` — do not hardcode values.
- Use type hints on all function signatures.
- Call `load_dotenv()` at the top of the file.
- **Do not** include `if __name__ == "__main__":` blocks in Prefect flow files.

### 3. Register the deployment

Edit `deploy_flows.py` to import and serve the new flow:

```python
from f03_my_pipeline import my_pipeline

if __name__ == "__main__":
    serve(
        # ... existing deployments ...
        my_pipeline.to_deployment(name="My Pipeline"),
    )
```

### 4. Re-deploy

Stop the running `deploy_flows.py` process (if any) and re-run:

```bash
uv run python deploy_flows.py
```

### 5. Add reusable utilities

If your flow introduces logic that could be shared across pipelines, place it in the `utils/` package rather than duplicating code.

