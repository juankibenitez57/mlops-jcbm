from dotenv import load_dotenv
load_dotenv()

import os

import requests
from prefect import flow, task
from prefect.logging import get_run_logger

from utils.config import load_config


@task(name="Launch RunPod pod")
def launch_runpod_pod() -> dict:
    logger = get_run_logger()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    runpod_api_key = os.getenv("RUNPOD_API_KEY")

    runpod_cfg = load_config("runpod")["training"]
    docker_cfg = load_config("docker")["training"]

    url = "https://rest.runpod.io/v1/pods"

    payload = {
        "computeType": runpod_cfg["compute_type"],
        "containerDiskInGb": runpod_cfg["container_disk_in_gb"],
        "env": {
            "WANDB_API_KEY": wandb_api_key,
            "RUNPOD_MANAGE_KEY": runpod_api_key, #Runpod injects a pod-scoped API Key called RUNPOD_API_KEY on each Pod
        },
        "imageName": docker_cfg["image_name"],
        "name": runpod_cfg["pod_name"],
        "volumeInGb": runpod_cfg["volume_in_gb"],
        "volumeMountPath": runpod_cfg["volume_mount_path"],
    }

    headers = {
        "Authorization": f"Bearer {runpod_api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    logger.info(f"RunPod response: {response.status_code}")
    return response.json()


@flow(name="Launch RunPod training pod")
def launch_runpod_pod_pipeline() -> None:
    result = launch_runpod_pod()
    logger = get_run_logger()
    logger.info(f"Pod launched: {result}")