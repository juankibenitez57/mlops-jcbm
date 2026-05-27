# Automation with Prefect

This project demonstrates an MLOps workflow using [Prefect](https://www.prefect.io/) to orchestrate the training and deployment of a KNN model on the Iris dataset. It integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking and artifact management, [Docker](https://www.docker.com/) for containerization, and [RunPod](https://www.runpod.io/) for cloud-based training and serverless inference.

## Project Structure

```
├── config/
│   ├── docker.json        # Docker image names (training & predict)
│   ├── runpod.json        # RunPod pod & serverless deployment settings
│   ├── training.json      # Model hyperparameters, features, train/test split
│   └── wandb.json         # W&B entity, project, artifact names
├── predict/
│   ├── Dockerfile         # Container image for serverless inference
│   ├── download_model.py  # Downloads trained model from W&B artifacts
│   ├── handler.py         # RunPod serverless handler
│   └── requirements.txt   # Predict container dependencies
├── train/
│   ├── Dockerfile         # Container image for the training job (RunPod)
│   └── train.py           # Model training script (used inside the container)
├── utils/
│   ├── __init__.py
│   ├── config.py          # JSON config loader utility
│   └── runpod/
│       ├── __init__.py
│       ├── update_endpoint.py  # Update RunPod serverless endpoint
│       └── update_template.py  # Update RunPod template image
├── deploy_flows.py        # Serves Prefect flow deployments
├── docker-compose.yml     # Prefect server infrastructure (Postgres, Redis, server, worker)
├── f01_build.py           # Flow: build/push Docker training image
├── f02_train_pod.py       # Flow: launch a RunPod training pod
├── f03_deploy.py          # Flow: build/push predict image & update serverless endpoint
└── pyproject.toml         # Python project metadata and dependencies
```

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** — must be installed and running
- A **Docker Hub** account (with a Personal Access Token stored in `DOCKER_PAT`)
- A **Weights & Biases** account (API key stored in `WANDB_API_KEY`)
- A **RunPod** account (API key stored in `RUNPOD_API_KEY`) — needed for cloud training and serverless deployment

## Environment Setup

1. **Install uv** (if not already installed):

   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and sync the virtual environment**:

   ```bash
   uv sync
   ```

   This reads `pyproject.toml`, creates a `.venv` in the project directory, and installs all dependencies (`prefect`, `scikit-learn`, `wandb`, `pandas`, etc.).

3. **Create a `.env` file** (or rename the .env.example file) in the project root with the required secrets:

   ```env
   PREFECT_API_URL=http://127.0.0.1:4200/api
   DOCKER_PAT=your_docker_hub_personal_access_token
   WANDB_API_KEY=your_wandb_api_key
   RUNPOD_API_KEY=your_runpod_api_key
   ```

## Starting Prefect

The Prefect server runs via Docker Compose and includes all background services, and a worker.

1. (if needed) **Ensure Docker Desktop is running.** Open Docker Desktop and verify the engine is active.

2. **Start the Prefect infrastructure**:

   ```bash
   docker compose up -d
   ```

   This spins up five services:

   | Service              | Purpose                            | Port  |
   |----------------------|------------------------------------|-------|
   | `postgres`           | Prefect metadata database          | —     |
   | `redis`              | Event messaging / cache            | —     |
   | `prefect-server`     | API server & UI                    | 4200  |
   | `prefect-services`   | Background services (scheduler…)   | —     |
   | `prefect-worker`     | Executes flow runs (`local-pool`)  | —     |

3. **Access the Prefect UI** at [http://localhost:4200](http://localhost:4200).

4. **Stop the infrastructure** when you're done:

   ```bash
   docker compose down
   ```

   Add `-v` to also remove the database and Redis volumes.

## Deploying Flows

The file `deploy_flows.py` serves the existing flow deployments so the Prefect worker can pick them up.

```bash
uv run python deploy_flows.py
```

This registers three deployments with the Prefect server:

| Deployment                           | Source File      | Description                                                              |
|--------------------------------------|------------------|--------------------------------------------------------------------------|
| **Build and push Docker training image** | `f01_build.py`   | Logs into Docker Hub, builds the training image, and pushes it           |
| **Launch RunPod training pod**       | `f02_train_pod.py` | Creates a RunPod pod that runs the training container                    |
| **Serverless Deployment**            | `f03_deploy.py`  | Downloads model, builds/pushes predict image, updates RunPod endpoint    |

Once deployed, you can trigger runs from the Prefect UI or via the CLI.

## Existing Flows

### `f01_build.py` — Build & Push Training Image

Authenticates with Docker Hub, builds the training container image for `linux/amd64` (using `train/Dockerfile`), and pushes it. Configured via `config/docker.json` → `training.image_name`.

### `f02_train_pod.py` — Launch RunPod Training Pod

Calls the RunPod REST API to launch a training pod using the built Docker image. Pod configuration (compute type, disk size, volume) comes from `config/runpod.json` → `training`.

### `f03_deploy.py` — Serverless Deployment

Handles the full deployment pipeline for serving predictions via RunPod Serverless:

1. Downloads the latest trained model from W&B artifacts into `predict/model/`
2. Authenticates with Docker Hub
3. Builds the predict container image (using `predict/Dockerfile`)
4. Pushes the predict image to Docker Hub
5. Updates the RunPod template with the new image
6. Updates the RunPod serverless endpoint to use the updated template

Configured via `config/docker.json` → `predict.image_name` and `config/runpod.json` → `deployment`.

### `train/train.py` — Training Script

Runs inside the Docker container on RunPod. It:

1. Downloads a processed dataset artifact from W&B
2. Splits data into train/test sets
3. Trains a KNN classifier
4. Evaluates the model and logs metrics (accuracy, precision, recall, F1) to W&B
5. Registers the trained model as a W&B artifact
6. Stops the RunPod pod after a 60-second grace period for log inspection

All parameters (features, target, test size, neighbors, etc.) are read from `config/training.json` and `config/wandb.json`.

### `predict/handler.py` — Serverless Inference Handler

RunPod serverless handler that loads the trained KNN model and serves predictions. Expects input in the format:

```json
{
  "input": {
    "samples": [
      {
        "SepalLengthCm": 5.1,
        "SepalWidthCm": 3.5,
        "PetalLengthCm": 1.4,
        "PetalWidthCm": 0.2,
        "PetalAreacm2": 0.28,
        "SepalAreacm2": 17.85
      }
    ]
  }
}
```

## Configuration

Project settings are split across individual JSON files in the `config/` directory:

| File              | Contents                                                       |
|-------------------|----------------------------------------------------------------|
| `docker.json`     | Docker image names for training and predict containers         |
| `runpod.json`     | Pod config (training) and serverless deployment config         |
| `training.json`   | Target column, features, train/test split ratio, KNN params    |
| `wandb.json`      | W&B entity, project name, artifact names and versions          |

Configuration is loaded using `load_config("name")` from `utils.config`, which resolves files from the `config/` directory (e.g., `load_config("wandb")` loads `config/wandb.json`).

## Creating New Flows

To add a new Prefect flow to the project:

1. **Create a new Python file** in the project root (e.g., `f04_evaluate.py`):

   ```python
   from dotenv import load_dotenv
   load_dotenv()

   from prefect import flow, task
   from prefect.logging import get_run_logger
   from utils.config import load_config


   @task(name="My new task")
   def my_task(config: dict) -> None:
       logger = get_run_logger()
       # your logic here
       logger.info("Task completed")


   @flow(name="My new flow")
   def my_new_flow() -> None:
       config = load_config("training")
       my_task(config)
   ```

2. **Register the deployment** by importing the flow in `deploy_flows.py`:

   ```python
   from f04_evaluate import my_new_flow

   if __name__ == "__main__":
       serve(
           build_and_push_pipeline.to_deployment(name="Build and push Docker training image"),
           launch_runpod_pod_pipeline.to_deployment(name="Launch RunPod training pod"),
           deploy_predict_pipeline.to_deployment(name="Serverless Deployment"),
           my_new_flow.to_deployment(name="My new flow"),
       )
   ```

3. **Restart the deployment server**:

   ```bash
   uv run python deploy_flows.py
   ```

4. **Add new dependencies** if your flow requires additional packages:

   ```bash
   uv add package-name
   ```

   This updates `pyproject.toml` and `uv.lock` automatically.

### Tips for Expanding the Project

- **Use `config/` files** for any parameters your flow needs — load them with `load_config("name")` from `utils.config`.
- **Use `@task`** for individual units of work and **`@flow`** to compose them. This gives you retry logic, logging, and observability out of the box.
- **Add schedules** to deployments if you want periodic execution:
  ```python
  my_new_flow.to_deployment(
      name="My scheduled flow",
      cron="0 8 * * *",  # every day at 08:00
  )
  ```
- **Modify `train/train.py`** to change the training logic (different model, hyperparameters, dataset). The training Dockerfile copies it into the container automatically.
- **Rebuild the Docker image** after changes to `train/` or `config/` by running the *Build and push Docker training image* flow.
- **Rebuild the predict image** after retraining by running the *Serverless Deployment* flow, which downloads the latest model and redeploys.

