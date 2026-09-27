"""server.py must activate the Agent before api.yaml_compat picks a YAML backend."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent

# PyYAML is always blocked; ruamel.yaml stays invisible until run_agent is imported,
# mirroring a managed runtime that exposes its dependencies on Agent import.
_GATE = (
    "import builtins, sys\n"
    "_orig = builtins.__import__\n"
    "def _gate(name, *a, **k):\n"
    "    top = name.split('.')[0]\n"
    "    if top == 'yaml' or (top == 'ruamel' and 'run_agent' not in sys.modules):\n"
    "        raise ImportError(f'{name} not activated')\n"
    "    return _orig(name, *a, **k)\n"
    "builtins.__import__ = _gate\n"
)


def test_server_import_activates_agent_before_ruamel_only_yaml(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "run_agent.py").write_text("class AIAgent: pass\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("HERMES_WEBUI_") and k != "PYTHONPATH"}
    env.update(
        HERMES_WEBUI_AGENT_DIR=str(agent_dir),
        HERMES_HOME=str(tmp_path / "home"),
        HERMES_BASE_HOME=str(tmp_path / "home"),
        HERMES_WEBUI_STATE_DIR=str(tmp_path / "state"),
    )
    script = _GATE + (
        "import server\n"
        "from api import yaml_compat\n"
        "assert sys.modules['run_agent'].__file__ == sys.argv[1], sys.modules['run_agent'].__file__\n"
        "assert yaml_compat.BACKEND == 'ruamel'\n"
        "assert yaml_compat.safe_load('a: 1') == {'a': 1}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(agent_dir / "run_agent.py")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
