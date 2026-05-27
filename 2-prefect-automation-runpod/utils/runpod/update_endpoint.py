import argparse
import os
import sys

import requests


def update_endpoint(endpoint_id: str, template_id: str) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY environment variable is not set")

    url = f"https://rest.runpod.io/v1/endpoints/{endpoint_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "templateId": template_id,
    }

    response = requests.patch(url, json=payload, headers=headers)
    response.raise_for_status()

    print(response.text)
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update a RunPod endpoint with a template")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--template-id", required=True, help="RunPod template ID to assign")
    args = parser.parse_args()

    print(f"Endpoint ID: {args.endpoint_id}")
    print(f"Template ID: {args.template_id}")

    update_endpoint(args.endpoint_id, args.template_id)