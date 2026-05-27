import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def load_config(name: str) -> dict:
    """Load a JSON config file from the config/ directory.

    *name* can be:
      - A bare name (e.g. ``"wandb"``) → resolves to ``config/wandb.json``
      - A name with ``.json`` suffix (e.g. ``"wandb.json"``) → same as above
      - An absolute path → loaded directly
    """
    p = Path(name)
    if p.is_absolute():
        return json.loads(p.read_text())
    stem = p.stem if p.suffix == ".json" else name
    return json.loads((_CONFIG_DIR / f"{stem}.json").read_text())
