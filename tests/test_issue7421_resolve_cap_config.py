"""Regression coverage for issue #7421: configurable
``HERMES_WEBUI_MAX_SESSION_RESOLVE`` cap on the heavy
full-transcript resolve semaphore.

The cap is hardcoded to ``2`` in
``api/models.py::_FULL_SESSION_RESOLVE_MAX_CONCURRENT``. The
maintainer's diagnosis is that the literal 2 is too low for
high-concurrency deployments where several parallel active
sessions all need a full-transcript resolve, and that the
deeper per-read cost is tracked separately in #7310. This PR
is the self-contained first step: read the cap from the env
var so operators can raise it without code changes, while
preserving the previous default of 2 and bounding the upper
end against typos like ``=999999``.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _import_models_with_env(env_value):
    """Reload ``api.models`` with a chosen env value for
    ``HERMES_WEBUI_MAX_SESSION_RESOLVE``. The cap is a module-level
    constant computed at import time, so the env var has to be
    set before the module is first imported."""
    if env_value is None:
        os.environ.pop("HERMES_WEBUI_MAX_SESSION_RESOLVE", None)
    else:
        os.environ["HERMES_WEBUI_MAX_SESSION_RESOLVE"] = env_value
    # Drop any cached import so the module-level constant is
    # recomputed against the freshly-set env.
    for mod_name in list(sys.modules):
        if mod_name == "api.models" or mod_name.startswith("api.models."):
            del sys.modules[mod_name]
    return importlib.import_module("api.models")


# ── default behavior (env unset) ──────────────────────────────────────────────


def test_default_cap_is_two_when_env_unset():
    """Backward compatibility: an installation that does not set
    the env var keeps the previous behavior. The bounded
    semaphore is constructed with capacity 2, so 3+ parallel
    full-resolves queue behind two slots as before."""
    models = _import_models_with_env(None)
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 2, (
        "unset env must keep the previous hardcoded default of 2 "
        "so existing single-instance deployments are unchanged"
    )


def test_default_cap_is_two_when_env_empty_string():
    """A user who sets the env var to an explicit empty string
    (e.g. from a misconfigured .env file) is the same as unset
    and must fall back to the default of 2 rather than 0."""
    models = _import_models_with_env("")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 2, (
        "empty string must be treated as unset, not as the "
        "literal value 0 (which would deadlock the semaphore)"
    )


def test_default_cap_is_two_when_env_whitespace_only():
    """A user who copies a trailing space from a config file or
    shell quoting must not accidentally deadlock the semaphore
    by setting the cap to 0."""
    models = _import_models_with_env("   ")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 2


# ── env override success cases ───────────────────────────────────────────────


def test_env_override_takes_effect_within_upper_bound():
    """The headline use case: a multi-instance operator with 4
    parallel active sessions sets the env to 4 and gets a
    bounded semaphore with capacity 4 instead of 2. This
    mirrors the maintainer's recommendation in the issue."""
    models = _import_models_with_env("4")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 4, (
        "operator intent must win when the env var is a positive "
        "int; the previous hardcoded 2 was a regression for "
        "the report's 4-parallel-session case"
    )


def test_env_override_takes_effect_at_upper_bound():
    """64 is the explicit safety net upper bound. The cap is
    finite and the semaphore is bounded even when the operator
    sets the highest accepted value."""
    models = _import_models_with_env("64")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 64


# ── env override failure / safety cases ──────────────────────────────────────


def test_env_zero_falls_back_to_default():
    """``HERMES_WEBUI_MAX_SESSION_RESOLVE=0`` would deadlock the
    bounded semaphore the first time it was acquired; an
    operator typo must not break the WebUI at runtime."""
    models = _import_models_with_env("0")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 2, (
        "0 must fall back to the default; a bounded semaphore "
        "with capacity 0 is a deadlock, not a concurrency cap"
    )


def test_env_negative_falls_back_to_default():
    """``HERMES_WEBUI_MAX_SESSION_RESOLVE=-1`` is just as
    dangerous as 0 (also a deadlock) and must be rejected
    silently with the same fallback."""
    models = _import_models_with_env("-1")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 2


def test_env_above_upper_bound_falls_back_to_default():
    """``HERMES_WEBUI_MAX_SESSION_RESOLVE=999999`` is almost
    certainly a typo and must NOT defeat the safety-bound
    design by clamping up to the ceiling. The previous
    clamp-to-64 behavior (round-1) admitted 64 simultaneous
    unbounded transcript loads; a synthetic 4.27 MB transcript
    consumed ~9.6 MB per parse, so the round-1 ceiling
    permitted ~614 MB at 64-way concurrency before any other
    application memory.

    The cap is a *safety bound*, not a user preference — a
    typo must fall back to the safe default (2), never to
    the permissive extreme. The upper bound (64) is reachable
    only via an explicit in-range operator value; out-of-range
    values degrade silently to the default, matching the
    behavior of every other bad input in this helper. See
    #7656 round-3 finding 1.
    """
    models = _import_models_with_env("999999")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 2, (
        "out-of-band values must fall back to the safe default, "
        "not clamp up to the ceiling; a typo should never widen "
        "the cap"
    )


def test_env_non_numeric_falls_back_to_default():
    """``HERMES_WEBUI_MAX_SESSION_RESOLVE=many`` is a config
    mistake and must not crash import. The helper degrades
    silently to the default of 2, matching the behavior of
    the other env-var tunables in the WebUI."""
    models = _import_models_with_env("many")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 2


def test_env_empty_after_strip_falls_back_to_default():
    """A user who pastes a value with leading/trailing whitespace
    (``'  3  '``) is accepted; a value that is *only*
    whitespace (``'   '``) is treated as unset and falls back
    to the default. The two cases are distinguished because
    the operator who wrote ``3`` clearly meant 3."""
    models = _import_models_with_env("  3  ")
    assert models._FULL_SESSION_RESOLVE_MAX_CONCURRENT == 3


# ── bounded-semaphore capacity is consistent with the cap ────────────────────


def test_bounded_semaphore_capacity_matches_cap():
    """The semaphore is constructed with the same int that
    ``_FULL_SESSION_RESOLVE_MAX_CONCURRENT`` was assigned, so
    raising the cap actually widens the concurrency budget.
    Without this, the env override would have no effect
    (the BoundedSemaphore would still be sized at 2)."""
    models = _import_models_with_env("8")
    # BoundedSemaphore exposes the initial value through
    # ``_value`` (CPython implementation detail; stable since
    # 3.x for the threading.BoundedSemaphore in the stdlib).
    sem = models._FULL_SESSION_RESOLVE_SLOTS
    capacity = getattr(sem, "_value", None)
    assert capacity == 8, (
        f"_FULL_SESSION_RESOLVE_SLOTS must be sized to the "
        f"resolved cap (8), got {capacity!r}; otherwise the "
        f"operator's env override has no effect"
    )


# ── static wiring ───────────────────────────────────────────────────────────


def test_helper_defined_in_models():
    """The helper itself must live next to the cap it configures
    so a future maintainer can find the precedence rules
    without searching the codebase. The helper must also
    accept the operator-facing env name verbatim — no alias
    without a comment."""
    src = (ROOT / "api" / "models.py").read_text(encoding="utf-8")
    assert "def _read_max_session_resolve_concurrent" in src
    assert "HERMES_WEBUI_MAX_SESSION_RESOLVE" in src, (
        "the env var name must appear in api/models.py so the "
        "operator-facing contract is discoverable from this "
        "module alone"
    )
    # The cap is now sourced from the helper rather than the
    # bare literal.
    assert "_FULL_SESSION_RESOLVE_MAX_CONCURRENT = _read_max_session_resolve_concurrent()" in src
    # The bare literal-2 initializer line is gone; a stray
    # duplicate would either keep the env-override silent or
    # confuse the next reader about which value wins.
    assert src.count("_FULL_SESSION_RESOLVE_MAX_CONCURRENT = 2\n") == 0, (
        "the bare literal-2 initializer must be removed; the "
        "cap is now sourced from the helper"
    )


# ── #7656 round-3 finding 2: profile .env must not resize the cap ───────────


def test_protected_env_keys_includes_resolve_cap():
    """``HERMES_WEBUI_MAX_SESSION_RESOLVE`` is a process-wide
    resource cap; a profile's ``.env`` must not override it.

    The operator/launcher env at startup is the only place the
    cap is configurable. Without this, a first-loaded profile
    could set the cap to one value and a later profile-switch
    would silently re-size the already-constructed
    BoundedSemaphore — the same first-load-wins asymmetry
    the maintainer flagged for the isolated-profile key in
    #4589. Pin the contract.
    """
    from api.profiles import _PROTECTED_ENV_KEYS
    assert "HERMES_WEBUI_MAX_SESSION_RESOLVE" in _PROTECTED_ENV_KEYS, (
        "HERMES_WEBUI_MAX_SESSION_RESOLVE must be in "
        "_PROTECTED_ENV_KEYS so a profile's .env cannot "
        "override the process-wide cap"
    )


def test_blocked_runtime_env_keys_includes_resolve_cap():
    """``filter_runtime_env_for_gateway_parity`` must also drop
    the resolve cap from a profile's runtime env, so the
    gateway/CLI/agent paths see the operator-set value rather
    than whatever the profile tried to set.

    This is the runtime counterpart of the protected-key
    check. Together they pin the contract from both sides:
    a profile cannot inject the env var at any layer
    (startup reload, runtime env, gateway parity filter).
    """
    from api.profiles import _BLOCKED_RUNTIME_ENV_KEYS
    assert "HERMES_WEBUI_MAX_SESSION_RESOLVE" in _BLOCKED_RUNTIME_ENV_KEYS, (
        "HERMES_WEBUI_MAX_SESSION_RESOLVE must be in "
        "_BLOCKED_RUNTIME_ENV_KEYS so a profile's runtime env "
        "is filtered out before the gateway/CLI/agent paths "
        "see it"
    )


def test_filter_runtime_env_strips_resolve_cap():
    """A profile env containing the resolve cap must be filtered
    out by ``filter_runtime_env_for_gateway_parity``.

    End-to-end exercise of the SILENT finding: the
    runtime-env filter is the one that actually prevents the
    cap from being silently re-sized by a profile switch.
    Without this, the cap the operator set at startup is
    silently overridden the first time a profile with the
    env var in its ``.env`` loads.
    """
    from api.profiles import filter_runtime_env_for_gateway_parity
    profile_env = {
        "HERMES_WEBUI_MAX_SESSION_RESOLVE": "64",
        "LANG": "en_US.UTF-8",
    }
    filtered = filter_runtime_env_for_gateway_parity(profile_env)
    assert "HERMES_WEBUI_MAX_SESSION_RESOLVE" not in filtered, (
        "filter_runtime_env_for_gateway_parity must strip "
        "HERMES_WEBUI_MAX_SESSION_RESOLVE from a profile's "
        "runtime env so the cap stays at the operator/launcher value"
    )
    # Sanity: an unrelated, non-blocked env key still passes through.
    # (PATH, HOME, etc. are themselves in _BLOCKED_RUNTIME_ENV_KEYS,
    # so pick something neutral like LANG.)
    assert filtered.get("LANG") == "en_US.UTF-8"

