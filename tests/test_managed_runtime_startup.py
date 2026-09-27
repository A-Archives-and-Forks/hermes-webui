"""The managed Agent must initialize before WebUI imports, without hiding api."""

import os
import subprocess
import sys

import pytest

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


@pytest.mark.parametrize("agent_source", [None, '"""No-Agent browser fixture."""\n'])
def test_server_starts_without_agent_class(tmp_path, agent_source):
    agent_dir = tmp_path / "no-agent"
    agent_dir.mkdir()
    if agent_source is not None:
        (agent_dir / "run_agent.py").write_text(agent_source, encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HERMES_WEBUI_AGENT_DIR"] = str(agent_dir)
    webui_root = os.path.dirname(os.path.dirname(__file__))
    # Stop at the first WebUI import: this exercises the real startup seam
    # without involving unrelated imports from any Agent installed on the host.
    script = (
        "import builtins\n"
        "original = builtins.__import__\n"
        "def check(name, *args, **kwargs):\n"
        "    if name == 'api.request_logging':\n"
        "        raise SystemExit(0)\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = check\n"
        "import server\n"
        "raise SystemExit('WebUI imports not reached')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=webui_root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_server_does_not_hide_agent_internal_import_failure(tmp_path):
    agent_dir = tmp_path / "broken-agent"
    agent_dir.mkdir()
    (agent_dir / "run_agent.py").write_text(
        "import deliberately_missing_agent_dependency\nclass AIAgent: pass\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HERMES_WEBUI_AGENT_DIR"] = str(agent_dir)
    result = subprocess.run(
        [sys.executable, "-c", "import server"],
        cwd=os.path.dirname(os.path.dirname(__file__)), env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "No module named 'deliberately_missing_agent_dependency'" in result.stderr
