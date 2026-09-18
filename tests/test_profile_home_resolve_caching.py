"""Regression test: `_resolve_profile_home_param` memoizes its filesystem
`.resolve()` call, but ONLY for the duration of a single
`profile_home_resolve_cache_scope()` (currently wrapping
`_load_cli_sessions_uncached`, the sidebar's session-list build).

Building the CLI session list calls `_resolve_profile_home_param(profile)`
once per session row (`Session.__init__`), so with hundreds of sessions on
the same profile the unscoped code redundantly re-resolved the identical
path via a filesystem `.resolve()` on every single row.

An earlier revision of this fix cached the result in a plain process-lifetime
module dict. That was flagged in PR #7636 review: `Path.resolve()` is a
filesystem call, not a pure function of process-startup constants -- a
profile-home symlink can be retargeted while the webui process keeps
running, and `_safe_resolve()` deliberately falls back to the UNRESOLVED
input after a transient error, so a process-lifetime cache could serve a
stale or wrong path for the rest of the process's life.

This file tests the corrected, call-scoped design:
  (a) within one scope, repeated resolves of the same argument hit the cache
      (single underlying `_safe_resolve()` call) -- same benefit as before.
  (b) across two SEPARATE scopes (simulating two different requests/list
      builds), the cache does NOT persist -- `_safe_resolve()` runs again
      fresh on the second scope, proving no cross-request staleness.
  (c) outside any scope, `_resolve_profile_home_param` resolves fresh every
      time with no caching at all, matching pre-PR behavior exactly.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.workspace as ws  # noqa: E402


def test_resolve_profile_home_param_caches_within_one_scope(tmp_path):
    """(a) Same as the original test, but the cache now requires an active scope."""
    home_dir = tmp_path / "profile_home"
    home_dir.mkdir()

    with patch.object(ws, "_safe_resolve", wraps=ws._safe_resolve) as spy:
        with ws.profile_home_resolve_cache_scope():
            first = ws._resolve_profile_home_param(home_dir)
            second = ws._resolve_profile_home_param(home_dir)
            third = ws._resolve_profile_home_param(home_dir)

    assert first == second == third == home_dir.resolve()
    assert spy.call_count == 1, (
        "expected the filesystem resolve() to run once within one scope and "
        f"be served from cache thereafter, but it ran {spy.call_count} times"
    )


def test_resolve_profile_home_param_caches_within_one_scope_for_profile_name(tmp_path):
    """(a), second branch: a logical profile-NAME STRING (not a Path) resolves

    through `get_hermes_home_for_profile()` before hitting the same
    `_cached_safe_resolve_profile_home()` memoization. Confirms the
    within-scope cache is genuinely hit for named profiles too, not just
    explicit Path callers.
    """
    home_dir = tmp_path / "named_profile_home"
    home_dir.mkdir()

    with patch("api.profiles.get_hermes_home_for_profile", return_value=home_dir), \
            patch.object(ws, "_safe_resolve", wraps=ws._safe_resolve) as spy:
        with ws.profile_home_resolve_cache_scope():
            first = ws._resolve_profile_home_param("myprofile")
            second = ws._resolve_profile_home_param("myprofile")
            third = ws._resolve_profile_home_param("myprofile")

    assert first == second == third == home_dir.resolve()
    assert spy.call_count == 1, (
        "expected the filesystem resolve() to run once within one scope and "
        f"be served from cache thereafter, but it ran {spy.call_count} times"
    )


def test_cache_does_not_persist_across_separate_scopes(tmp_path):
    """(b) Two separate scoped calls (simulating two different requests) must

    NOT share a cache -- the whole point of the PR #7636 review fix. If a
    symlink were retargeted between the two "requests", the second scope
    must be free to observe the new target; that's only possible if
    `_safe_resolve()` is invoked again fresh on scope #2.
    """
    home_dir = tmp_path / "profile_home"
    home_dir.mkdir()

    with patch.object(ws, "_safe_resolve", wraps=ws._safe_resolve) as spy:
        with ws.profile_home_resolve_cache_scope():
            ws._resolve_profile_home_param(home_dir)
            ws._resolve_profile_home_param(home_dir)
        assert spy.call_count == 1

        # Second, separate scope -- simulates a later, independent request.
        with ws.profile_home_resolve_cache_scope():
            ws._resolve_profile_home_param(home_dir)
            ws._resolve_profile_home_param(home_dir)

    assert spy.call_count == 2, (
        "expected the second scope to re-resolve fresh (no cross-scope/"
        f"cross-request caching), but _safe_resolve ran {spy.call_count} times total"
    )


def test_no_caching_outside_any_scope(tmp_path):
    """(c) Outside a scope, every call resolves fresh -- matching pre-PR

    behavior exactly, since most call sites of `_resolve_profile_home_param`
    are NOT the sidebar list-build hot loop and must never see a cached
    (possibly stale) result.
    """
    home_dir = tmp_path / "profile_home"
    home_dir.mkdir()

    with patch.object(ws, "_safe_resolve", wraps=ws._safe_resolve) as spy:
        first = ws._resolve_profile_home_param(home_dir)
        second = ws._resolve_profile_home_param(home_dir)
        third = ws._resolve_profile_home_param(home_dir)

    assert first == second == third == home_dir.resolve()
    assert spy.call_count == 3, (
        "expected every call outside an active scope to resolve fresh with "
        f"no caching, but _safe_resolve ran {spy.call_count} times for 3 calls"
    )


def test_scope_decorator_use_matches_load_cli_sessions_uncached_wiring():
    """Sanity check that `profile_home_resolve_cache_scope()` is usable as a

    decorator (the way it wraps `_load_cli_sessions_uncached` in
    `api/models.py`), and that the ContextVar is active only during the
    decorated call.
    """
    assert ws._PROFILE_HOME_RESOLVE_SCOPE.get() is None

    @ws.profile_home_resolve_cache_scope()
    def _inside():
        return ws._PROFILE_HOME_RESOLVE_SCOPE.get()

    cache_during_call = _inside()
    assert cache_during_call is not None
    assert ws._PROFILE_HOME_RESOLVE_SCOPE.get() is None
