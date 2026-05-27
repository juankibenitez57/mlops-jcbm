from dotenv import load_dotenv
load_dotenv()

import os
import subprocess

from prefect import flow, task
from prefect.logging import get_run_logger

from utils.config import load_config


@task(name="Docker login")
def docker_login(image_name: str) -> None:
    logger = get_run_logger()
    username = image_name.split("/")[0]
    password = os.environ["DOCKER_PAT"]
    subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin"],
        input=password,
        text=True,
        check=True,
    )
    logger.info("Docker login successful")


@task(name="Build Docker image")
def docker_build(image_name: str) -> None:
    logger = get_run_logger()
    subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-t", image_name, "-f", "./train/Dockerfile", "."],
        check=True,
    )
    logger.info(f"Built Docker image '{image_name}'")


@task(name="Push Docker image")
def docker_push(image_name: str) -> None:
    logger = get_run_logger()
    subprocess.run(
        ["docker", "push", image_name],
        check=True,
    )
    logger.info(f"Pushed Docker image '{image_name}'")


@flow(name="Build and push Docker training image")
def build_and_push_pipeline() -> None:
    image_name = load_config("docker")["training"]["image_name"]
    docker_login(image_name)
    docker_build(image_name)
    docker_push(image_name)