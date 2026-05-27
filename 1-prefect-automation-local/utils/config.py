import json
from pathlib import Path


def load_config(name: str) -> dict:
    return json.loads(Path(f"config/{name}.json").read_text())
