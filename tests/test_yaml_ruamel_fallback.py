"""api.yaml_compat must work when PyYAML is absent (Hermes Agent's managed runtime ships ruamel only)."""

import builtins
import importlib
import io
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

CFG = {
    "model": {"default": "claude-opus", "provider": "anthropic"},
    "toolsets": ["web", "terminal"],
    "webui_chat_backend": "gateway",
    "webui_gateway_use_runs_api": True,
    "name": "caf\u00e9",
    "count": 3,
    "empty": None,
}


@pytest.fixture
def ruamel_compat(monkeypatch):
    pytest.importorskip("ruamel.yaml")
    real_import = builtins.__import__

    def _no_pyyaml(name, *args, **kwargs):
        if name == "yaml" or name.startswith("yaml."):
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m == "yaml" or m.startswith("yaml.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.delitem(sys.modules, "api.yaml_compat", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_pyyaml)
    mod = importlib.import_module("api.yaml_compat")
    yield mod
    sys.modules.pop("api.yaml_compat", None)


def test_ruamel_backend_selected_without_pyyaml(ruamel_compat):
    assert ruamel_compat.BACKEND == "ruamel"


def test_ruamel_round_trip(ruamel_compat):
    text = ruamel_compat.safe_dump(CFG, sort_keys=False, allow_unicode=True)
    assert ruamel_compat.safe_load(text) == CFG
    assert "caf\u00e9" in text
    assert ruamel_compat.safe_load(io.StringIO(text)) == CFG
    assert ruamel_compat.safe_load("") is None


def test_ruamel_dump_to_stream(ruamel_compat):
    buf = io.StringIO()
    assert ruamel_compat.dump(CFG, buf, default_flow_style=False, allow_unicode=True) is None
    assert ruamel_compat.safe_load(buf.getvalue()) == CFG


def test_ruamel_output_matches_pyyaml(ruamel_compat):
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, json, yaml; print(yaml.safe_dump(json.loads(sys.argv[1]), sort_keys=False, allow_unicode=True), end='')",
         __import__("json").dumps(CFG)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        pytest.skip("PyYAML not installed in test interpreter")
    assert ruamel_compat.safe_dump(CFG, sort_keys=False, allow_unicode=True) == out.stdout


def test_config_loader_reads_yaml_without_pyyaml(ruamel_compat, tmp_path, monkeypatch):
    import api.onboarding as onboarding

    cfg = tmp_path / "config.yaml"
    cfg.write_text(ruamel_compat.safe_dump(CFG), encoding="utf-8")
    assert onboarding._load_yaml_config(cfg) == CFG


def test_no_bare_pyyaml_imports_in_server_code():
    offenders = []
    for path in [REPO / "server.py", *sorted((REPO / "api").glob("*.py"))]:
        if path.name == "yaml_compat.py":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith(("import yaml", "from yaml ")):
                offenders.append(f"{path.relative_to(REPO)}:{n}: {s}")
    assert offenders == [], "import YAML via api.yaml_compat:\n" + "\n".join(offenders)


def test_bootstrap_probe_accepts_ruamel_only():
    import bootstrap

    src = Path(bootstrap.__file__).read_text(encoding="utf-8")
    start = src.index("def _python_can_run_webui_and_agent")
    body = src[start:start + 600]
    assert "import ruamel.yaml" in body
    assert body.index("from run_agent import AIAgent") < body.index("import yaml")
