from dotenv import load_dotenv
load_dotenv()

import os

import runpod
from prefect import flow, task
from prefect.logging import get_run_logger

from utils.config import load_config


@task(name="Launch RunPod pod")
def launch_runpod_pod() -> dict:
    logger = get_run_logger()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    runpod_api_key = os.getenv("RUNPOD_API_KEY")

    runpod.api_key = runpod_api_key
    runpod_cfg = load_config("runpod")["training"]
    docker_cfg = load_config("docker")["training"]

    pod = runpod.create_pod(
        name=runpod_cfg["pod_name"],
        image_name=docker_cfg["image_name"],
        gpu_type_id=runpod_cfg["gpu_type_id"],
        cloud_type="ALL",
        container_disk_in_gb=runpod_cfg["container_disk_in_gb"],
        volume_in_gb=runpod_cfg["volume_in_gb"],
        volume_mount_path=runpod_cfg["volume_mount_path"],
        env={
            "WANDB_API_KEY": wandb_api_key,
            "RUNPOD_MANAGE_KEY": runpod_api_key,
        },
    )
    logger.info(f"Pod launched: id={pod['id']}, gpu={runpod_cfg['gpu_type_id']}")
    return pod


@flow(name="Launch RunPod training pod")
def launch_runpod_pod_pipeline() -> None:
    result = launch_runpod_pod()
    logger = get_run_logger()
    logger.info(f"Pod launched: {result}")


if __name__ == "__main__":
    launch_runpod_pod_pipeline()