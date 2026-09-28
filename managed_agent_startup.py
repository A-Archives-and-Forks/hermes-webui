"""Initialize a discovered Agent source checkout before importing WebUI modules."""

import importlib
import os
from pathlib import Path
import sys


def activate_managed_agent() -> None:
    agent_dir = os.environ.get("HERMES_WEBUI_AGENT_DIR")
    if not agent_dir:
        return
    # Browser-only setups may deliberately point at an empty Agent directory.
    if not (Path(agent_dir) / "run_agent.py").is_file():
        return

    webui_root = str(Path(__file__).resolve().parent)
    if webui_root not in sys.path:
        sys.path.insert(0, webui_root)
    # Bootstrap's probe adds the checkout to its own PYTHONPATH, not ours.
    if agent_dir not in sys.path:
        sys.path.insert(1, agent_dir)
    # Activate dependencies without importing the application: api.config must
    # select the active profile before Agent modules cache profile-sensitive paths.
    # Older Agents and browser-only shims have no bootstrap layer to activate.
    if (Path(agent_dir) / "hermes_bootstrap.py").is_file():
        importlib.import_module("hermes_bootstrap")
