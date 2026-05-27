import argparse
import os
import sys

import requests


def update_template(template_id: str, image_name: str, container_disk_in_gb: int = 5) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY environment variable is not set")

    url = f"https://rest.runpod.io/v1/templates/{template_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "containerDiskInGb": container_disk_in_gb,
        "imageName": image_name,
    }

    response = requests.patch(url, json=payload, headers=headers)
    response.raise_for_status()

    print(response.text)
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update a RunPod template image tag")
    parser.add_argument("--template-id", required=True, help="RunPod template ID")
    parser.add_argument("--image-name", required=True, help="Full Docker image name with tag")
    args = parser.parse_args()

    print(f"Template ID: {args.template_id}")
    print(f"Image Name: {args.image_name}")

    update_template(args.template_id, args.image_name)
