"""The managed Agent must initialize before WebUI imports, without hiding api."""

import os
import subprocess
import sys


def test_server_keeps_api_importable_after_agent_hardens_sys_path(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "run_agent.py").write_text(
        "import sys\n"
        "sys.path[:] = [path for path in sys.path if path != '']\n"
        "class AIAgent: pass\n",
        encoding="utf-8",
    )
    webui_root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(agent_dir)
    env["HERMES_WEBUI_AGENT_DIR"] = str(agent_dir)
    # Import the actual server module, rather than mocking the api import or
    # relying on pytest's sys.path (which already contains the repository).
    script = (
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('server', 'server.py')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert 'run_agent' in __import__('sys').modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=webui_root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
