"""A retained per-profile models disk snapshot must be rejected after a switch
when a catalog source changed in the target profile: its ``.env`` provider
credentials, its installed model-provider plugins, or the profile itself
(delete + recreate with the same name).

Each test saves a real snapshot for profile ``demo`` through
``_save_models_cache_to_disk()``, changes one source while ``default`` is
active, performs the per-client switch exactly as ``/api/profile/switch`` does
(``invalidate_models_cache(delete_disk=False)``), and asserts the first
``get_available_models()`` for ``demo`` serves the fresh rebuild.
"""

from __future__ import annotations

import os
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


def _catalog(label: str) -> dict:
    return {
        "active_provider": "openai",
        "default_model": label,
        "configured_model_badges": {},
        "groups": [
            {
                "provider": "OpenAI",
                "provider_id": "openai",
                "models": [{"id": label, "label": label, "supports_fast_tier": False}],
            }
        ],
        "aliases": {},
    }


@pytest.fixture
def two_profiles(tmp_path: Path, monkeypatch):
    import api.config as cfg
    import api.profiles as profiles

    monkeypatch.delenv("HERMES_CONFIG_PATH", raising=False)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / "config.yaml").write_text("model:\n  default: default-model\n", encoding="utf-8")
    demo_home = default_home / "profiles" / "demo"
    demo_home.mkdir(parents=True)
    (demo_home / "config.yaml").write_text("model:\n  default: demo-model\n", encoding="utf-8")
    os.utime(demo_home / "config.yaml", (1_000_000, 1_000_000))

    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", default_home)
    monkeypatch.setattr(profiles, "_active_profile", "default")
    monkeypatch.setattr(profiles._tls, "profile", None, raising=False)
    monkeypatch.setattr(cfg, "_models_cache_path", tmp_path / "models_cache.json")
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_invoke_models_rebuild", lambda _builder: _catalog("fresh-model"))
    for attr in ("_cfg_mtime", "_cfg_path", "_cfg_fingerprint"):
        monkeypatch.setattr(cfg, attr, getattr(cfg, attr), raising=False)
    saved_cfg = dict(cfg._cfg_cache)
    cfg.invalidate_models_cache(delete_disk=False)
    cfg.reload_config()
    yield SimpleNamespace(cfg=cfg, profiles=profiles, demo_home=demo_home, cache=tmp_path / "models_cache.demo.json")
    profiles.clear_request_profile()
    cfg.invalidate_models_cache(delete_disk=False)
    cfg._cfg_cache.clear()
    cfg._cfg_cache.update(saved_cfg)


def _save_demo_snapshot(env) -> None:
    env.profiles.set_request_profile("demo")
    try:
        env.cfg._save_models_cache_to_disk(_catalog("snapshot-model"))
    finally:
        env.profiles.clear_request_profile()
    assert env.cache.exists()


def _switch_to_demo_and_fetch(env) -> dict:
    env.profiles.switch_profile("demo", process_wide=False)
    env.cfg.invalidate_models_cache(delete_disk=False)
    env.profiles.set_request_profile("demo")
    try:
        return env.cfg.get_available_models()
    finally:
        env.profiles.clear_request_profile()


def _write_plugin(root: Path, name: str, version: str, *, flat: bool) -> Path:
    base = root / "plugins" if flat else root / "plugins" / "model-providers"
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.yaml").write_text(
        f"name: {name}\nkind: model-provider\nversion: {version}\n", encoding="utf-8"
    )
    return d


def test_unchanged_sources_keep_serving_the_retained_snapshot(two_profiles):
    """Control: with no source change the switch reuses the snapshot (the speed-up)."""
    _save_demo_snapshot(two_profiles)
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "snapshot-model"


def test_switch_after_adding_env_key_rejects_snapshot(two_profiles):
    _save_demo_snapshot(two_profiles)
    (two_profiles.demo_home / ".env").write_text("DEEPSEEK_API_KEY=sk-new\n", encoding="utf-8")
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


def test_switch_after_removing_env_key_rejects_snapshot(two_profiles):
    env_file = two_profiles.demo_home / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-old\nOTHER=1\n", encoding="utf-8")
    _save_demo_snapshot(two_profiles)
    env_file.write_text("OTHER=1\n", encoding="utf-8")
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


def test_switch_after_export_prefix_rejects_snapshot(two_profiles):
    """`export KEY=` is key `export KEY` to the profile .env loaders, so the provider key is gone."""
    env_file = two_profiles.demo_home / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-old\n", encoding="utf-8")
    _save_demo_snapshot(two_profiles)
    env_file.write_text("export DEEPSEEK_API_KEY=sk-old\n", encoding="utf-8")
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


def test_env_fingerprint_keys_match_provider_env_loader(two_profiles):
    from api.providers import _load_env_file

    env_file = two_profiles.demo_home / ".env"
    env_file.write_text("export A=1\nB='x'\nC=\n# D=1\n", encoding="utf-8")
    loaded = sorted(k for k, v in _load_env_file(env_file).items() if v)
    assert two_profiles.cfg._models_cache_env_fingerprint(env_file)["present_keys"] == loaded


def test_env_value_rotation_keeps_snapshot_and_fingerprint_has_no_secret(two_profiles):
    env_file = two_profiles.demo_home / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-secret-one\n", encoding="utf-8")
    _save_demo_snapshot(two_profiles)
    assert "sk-secret-one" not in two_profiles.cache.read_text(encoding="utf-8")
    env_file.write_text("DEEPSEEK_API_KEY=sk-secret-two\n", encoding="utf-8")
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "snapshot-model"


@pytest.mark.parametrize("flat", [False, True], ids=["model-providers-dir", "flat-install-dir"])
def test_switch_after_installing_plugin_rejects_snapshot(two_profiles, flat):
    _save_demo_snapshot(two_profiles)
    _write_plugin(two_profiles.demo_home, "acme", "1.0.0", flat=flat)
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


def test_switch_after_removing_plugin_rejects_snapshot(two_profiles):
    plugin = _write_plugin(two_profiles.demo_home, "acme", "1.0.0", flat=False)
    _save_demo_snapshot(two_profiles)
    shutil.rmtree(plugin)
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


def test_switch_after_plugin_version_change_rejects_snapshot(two_profiles):
    _write_plugin(two_profiles.demo_home, "acme", "1.0.0", flat=True)
    _save_demo_snapshot(two_profiles)
    _write_plugin(two_profiles.demo_home, "acme", "2.0.0", flat=True)
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


def test_delete_profile_removes_its_models_cache(two_profiles, monkeypatch):
    import api.profiles as profiles

    monkeypatch.setattr(profiles, "_is_isolated_profile_mode", lambda: False)
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", None)
    _save_demo_snapshot(two_profiles)
    profiles.delete_profile_api("demo")
    assert not two_profiles.demo_home.exists()
    assert not two_profiles.cache.exists()


def test_recreate_same_name_profile_does_not_resurrect_old_catalog(two_profiles, monkeypatch):
    import api.profiles as profiles

    monkeypatch.setattr(profiles, "_is_isolated_profile_mode", lambda: False)
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", None)
    monkeypatch.setattr(profiles, "list_profiles_api", lambda: [])
    _save_demo_snapshot(two_profiles)
    stale = two_profiles.cache.read_text(encoding="utf-8")
    profiles.delete_profile_api("demo")
    # A snapshot left behind by an older build or another process.
    two_profiles.cache.write_text(stale, encoding="utf-8")
    profiles.create_profile_api("demo")
    assert not two_profiles.cache.exists()
    (two_profiles.demo_home / "config.yaml").write_text("model:\n  default: demo-model\n", encoding="utf-8")
    os.utime(two_profiles.demo_home / "config.yaml", (1_000_000, 1_000_000))
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


@pytest.mark.parametrize("flat", [False, True], ids=["model-providers-dir", "flat-install-dir"])
def test_switch_after_plugin_code_change_without_version_bump_rejects_snapshot(two_profiles, flat):
    """fallback_models lives in the plugin's code, which the loader imports; the version may not move."""
    plugin = _write_plugin(two_profiles.demo_home, "acme", "1.0.0", flat=flat)
    init = plugin / "__init__.py"
    init.write_text("FALLBACK_MODELS = ('acme-1',)\n", encoding="utf-8")
    os.utime(init, (1_000_000, 1_000_000))
    _save_demo_snapshot(two_profiles)
    init.write_text("FALLBACK_MODELS = ('acme-1', 'acme-2')\n", encoding="utf-8")
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


def test_switch_after_plugin_submodule_change_rejects_snapshot(two_profiles):
    plugin = _write_plugin(two_profiles.demo_home, "acme", "1.0.0", flat=False)
    (plugin / "__init__.py").write_text("from .models import FALLBACK_MODELS\n", encoding="utf-8")
    sub = plugin / "models.py"
    sub.write_text("FALLBACK_MODELS = ('acme-1',)\n", encoding="utf-8")
    os.utime(sub, (1_000_000, 1_000_000))
    _save_demo_snapshot(two_profiles)
    sub.write_text("FALLBACK_MODELS = ('acme-1', 'acme-2')\n", encoding="utf-8")
    assert _switch_to_demo_and_fetch(two_profiles)["default_model"] == "fresh-model"


def test_plugin_bytecode_cache_does_not_churn_fingerprint(two_profiles):
    plugin = _write_plugin(two_profiles.demo_home, "acme", "1.0.0", flat=False)
    (plugin / "__init__.py").write_text("X = 1\n", encoding="utf-8")
    before = two_profiles.cfg._models_cache_plugin_fingerprint(two_profiles.demo_home)
    (plugin / "__pycache__").mkdir()
    (plugin / "__pycache__" / "__init__.cpython-311.pyc").write_bytes(b"x")
    assert two_profiles.cfg._models_cache_plugin_fingerprint(two_profiles.demo_home) == before
