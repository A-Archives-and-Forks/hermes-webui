"""The managed Agent must initialize before WebUI imports, without hiding api."""

import os
import subprocess
import sys

import bootstrap


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
    assert bootstrap._python_can_run_webui_and_agent(sys.executable, agent_dir)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HERMES_WEBUI_AGENT_DIR"] = str(agent_dir)
    # A script launch starts in the Agent directory but puts the WebUI script
    # directory, not cwd, on sys.path. The probe's PYTHONPATH is not inherited.
    script = (
        "import importlib.util, sys\n"
        "sys.path[0] = " + repr(webui_root) + "\n"
        "spec = importlib.util.spec_from_file_location('server', "
        + repr(os.path.join(webui_root, "server.py")) + ")\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert 'run_agent' in sys.modules\n"
        "assert sys.modules['run_agent'].__file__ == "
        + repr(str(agent_dir / "run_agent.py")) + "\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=agent_dir, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
