from dotenv import load_dotenv
load_dotenv()

import os
import subprocess

from prefect import flow, task
from prefect.logging import get_run_logger

from utils.config import load_config
from predict.download_model import download_model as _download_model
from utils.runpod.update_template import update_template as _update_template
from utils.runpod.update_endpoint import update_endpoint as _update_endpoint


@task(name="Download model from W&B")
def download_model() -> None:
    logger = get_run_logger()
    _download_model()
    logger.info("Model downloaded to predict/model/")


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


@task(name="Build predict Docker image")
def docker_build_predict(image_name: str) -> None:
    logger = get_run_logger()
    subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-t", image_name, "./predict/"],
        check=True,
    )
    logger.info(f"Built Docker image '{image_name}'")


@task(name="Push predict Docker image")
def docker_push_predict(image_name: str) -> None:
    logger = get_run_logger()
    subprocess.run(
        ["docker", "push", image_name],
        check=True,
    )
    logger.info(f"Pushed Docker image '{image_name}'")


@task(name="Update RunPod template")
def update_template() -> None:
    logger = get_run_logger()
    deploy_cfg = load_config("runpod")["deployment"]
    image_name = load_config("docker")["predict"]["image_name"]
    result = _update_template(
        deploy_cfg["template_id"], image_name, deploy_cfg["container_disk_in_gb"]
    )
    logger.info(f"Template updated: {result}")


@task(name="Update RunPod endpoint")
def update_endpoint() -> None:
    logger = get_run_logger()
    deploy_cfg = load_config("runpod")["deployment"]
    result = _update_endpoint(deploy_cfg["endpoint_id"], deploy_cfg["template_id"])
    logger.info(f"Endpoint updated: {result}")


@flow(name="Serverless Deployment")
def deploy_predict_pipeline() -> None:
    image_name = load_config("docker")["predict"]["image_name"]
    download_model()
    docker_login(image_name)
    docker_build_predict(image_name)
    docker_push_predict(image_name)
    update_template()
    update_endpoint()


if __name__ == "__main__":
    deploy_predict_pipeline()
