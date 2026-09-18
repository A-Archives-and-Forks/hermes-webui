"""Regression test: `_resolve_profile_home_param` memoizes its filesystem
`.resolve()` call per profile argument (N+1 sidebar-hang fix).

Building the CLI session list calls `_resolve_profile_home_param(profile)`
once per session row (`Session.__init__`), so with hundreds of sessions on
the same profile the old code redundantly re-resolved the identical path via
a filesystem `.resolve()` on every single row. This asserts repeated calls
with the same profile argument return an equal Path while invoking the
underlying `_safe_resolve()` filesystem call only once.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.workspace as ws  # noqa: E402


def test_resolve_profile_home_param_caches_repeated_path_argument(tmp_path):
    ws._PROFILE_HOME_RESOLVE_CACHE.clear()
    home_dir = tmp_path / "profile_home"
    home_dir.mkdir()

    with patch.object(ws, "_safe_resolve", wraps=ws._safe_resolve) as spy:
        first = ws._resolve_profile_home_param(home_dir)
        second = ws._resolve_profile_home_param(home_dir)
        third = ws._resolve_profile_home_param(home_dir)

    assert first == second == third == home_dir.resolve()
    assert spy.call_count == 1, (
        "expected the filesystem resolve() to run once and be served from "
        f"cache thereafter, but it ran {spy.call_count} times"
    )


def test_resolve_profile_home_param_caches_repeated_profile_name_argument(tmp_path):
    """Same fix, second branch: a logical profile-NAME STRING (not a Path)

    resolves through `get_hermes_home_for_profile()` before hitting the same
    `_cached_safe_resolve_profile_home()` memoization. This mirrors the
    Path-argument test above but drives the string branch (workspace.py's
    ``isinstance(profile, Path)`` check is False), confirming the cache is
    genuinely hit -- not just for explicit Path callers -- and that the
    resolved home is correct.
    """
    ws._PROFILE_HOME_RESOLVE_CACHE.clear()
    home_dir = tmp_path / "named_profile_home"
    home_dir.mkdir()

    with patch("api.profiles.get_hermes_home_for_profile", return_value=home_dir), \
            patch.object(ws, "_safe_resolve", wraps=ws._safe_resolve) as spy:
        first = ws._resolve_profile_home_param("myprofile")
        second = ws._resolve_profile_home_param("myprofile")
        third = ws._resolve_profile_home_param("myprofile")

    assert first == second == third == home_dir.resolve()
    assert spy.call_count == 1, (
        "expected the filesystem resolve() to run once and be served from "
        f"cache thereafter, but it ran {spy.call_count} times"
    )
