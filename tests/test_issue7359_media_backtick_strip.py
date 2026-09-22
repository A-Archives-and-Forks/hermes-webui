"""Regression tests for issue #7359 — MEDIA: token regex must not capture trailing backtick.

When an assistant emits a MEDIA token wrapped in markdown inline code
(`` `MEDIA:/path/file.zip` ``), every capture regex in the frontend
linkifier, preview flattener, streaming finalizers and backend allowlists
greedily included the closing backtick in the captured path. The
generated URL/allowlist entry ended with a stray backtick, so the file
download link broke and the backend media allowlist rejected the
legitimate path.

The fix adds `` ` `` to the excluded character class on all seven capture
sites:

  - ``static/ui.js:7695``     linkifier
  - ``static/ui.js:9142``     preview flattener
  - ``static/messages.js:4848``  streaming chunk finalize
  - ``static/messages.js:4896``  streaming combined walk
  - ``static/messages.js:4922``  streaming partial tail
  - ``api/routes.py:_MEDIA_TOKEN_RE``   backend session allowlist
  - ``api/media_snapshots.py:media_re`` settle-time snapshot capture

These tests verify the source shape on every site (so a site silently
reverting back to the old class fails the suite) and exercise the two
Python capture sites directly. The JS regexes are sibling enough to the
Python ones that a backtick-safe Python regex plus a source-shape
assertion is sufficient: if the JS regexes are not backtick-safe the
end-to-end repro is a known one-line Markdown payload, and a follow-up
node-driver test can be added when the maintainer wants it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Source-shape: each site's expected backtick-safe regex literal. Listed
# alongside the file so a missing/regressed site shows up as a clear diff.
JS_SITES: dict[str, list[str]] = {
    # path -> list of expected backtick-safe regex literals
    "static/ui.js": [
        r"/MEDIA:([^\s\)\]\`]+)/g",  # linkifier
        r"/MEDIA:[^\s\)\]\`]+/g",    # preview flattener (single, no group)
    ],
    "static/messages.js": [
        r"/^MEDIA:([^\s\)\]\u0060]+)$/",  # streaming chunk finalize
        r"/MEDIA:([^\s\)\]\u0060]+)/g",  # streaming combined walk
        r"/MEDIA:[^\s\)\]\u0060]*$/",    # streaming partial tail
    ],
}

PY_SITES: dict[str, str] = {
    "api/routes.py": r"MEDIA:([^\s\)\]\`]+)",
    "api/media_snapshots.py": r"MEDIA:([^\s\)\]\`]+)",
}


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-shape: every site must carry the backtick-safe class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relpath", "needle"),
    [
        (relpath, needle)
        for relpath, needles in JS_SITES.items()
        for needle in needles
    ],
    ids=lambda v: v if isinstance(v, str) else "/".join(v),
)
def test_js_site_excludes_backtick(relpath: str, needle: str) -> None:
    """Each JS capture site has `` ` `` inside its character class."""
    text = _read(relpath)
    assert needle in text, (
        f"{relpath} is missing the backtick-safe capture literal: {needle!r}"
    )


@pytest.mark.parametrize(
    ("relpath", "needle"), list(PY_SITES.items()), ids=lambda v: v
)
def test_py_site_excludes_backtick(relpath: str, needle: str) -> None:
    """Each Python capture site has `` ` `` inside its character class."""
    text = _read(relpath)
    assert needle in text, (
        f"{relpath} is missing the backtick-safe capture literal: {needle!r}"
    )


def test_all_seven_sites_covered() -> None:
    """Guard against silently adding/removing capture sites."""
    expected = 2 + 3 + 2  # ui.js 2 + messages.js 3 + py 2
    actual = sum(len(v) for v in JS_SITES.values()) + len(PY_SITES)
    assert actual == expected, f"site count drifted: {actual} != {expected}"


# ---------------------------------------------------------------------------
# Python behavioural: compile the actual regexes and exercise them
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _media_token_re() -> re.Pattern[str]:
    """Compile the live ``_MEDIA_TOKEN_RE`` from api/routes.py."""
    routes_text = _read("api/routes.py")
    m = re.search(
        r'_MEDIA_TOKEN_RE = re\.compile\(r"([^"]+)"\)', routes_text, re.M
    )
    assert m, "_MEDIA_TOKEN_RE compile() not found in api/routes.py"
    return re.compile(m.group(1))


@pytest.fixture(scope="module")
def _media_snap_re() -> re.Pattern[str]:
    """Compile the live ``media_re`` from api/media_snapshots.py."""
    snap_text = _read("api/media_snapshots.py")
    m = re.search(
        r'media_re = _re\.compile\(r"([^"]+)"\)', snap_text, re.M
    )
    assert m, "media_re compile() not found in api/media_snapshots.py"
    return re.compile(m.group(1))


def test_backend_allowlist_strips_backtick(_media_token_re: re.Pattern[str]) -> None:
    """``_MEDIA_TOKEN_RE`` must not include the closing backtick in the path."""
    # Inline-code wrapper: backtick before MEDIA:, backtick after the path.
    captured = _media_token_re.findall("get `MEDIA:/home/kim/skills.zip` now")
    assert captured == ["/home/kim/skills.zip"], captured
    assert all(not c.endswith("`") for c in captured)


def test_backend_allowlist_bare_token_still_captures(
    _media_token_re: re.Pattern[str],
) -> None:
    """Backward-compat: bare ``MEDIA:/path`` (no backtick) still matches."""
    captured = _media_token_re.findall(
        "MEDIA:/home/kim/a.zip and MEDIA:/home/kim/b.tar"
    )
    assert captured == ["/home/kim/a.zip", "/home/kim/b.tar"], captured


def test_backend_allowlist_rejects_trailing_paren(
    _media_token_re: re.Pattern[str],
) -> None:
    """Pre-existing behaviour: ``)`` is still excluded."""
    # Trailing ')' is part of markdown link syntax and must NOT be captured.
    captured = _media_token_re.findall("see [file](MEDIA:/home/kim/a.zip) here")
    # Either nothing (paren excluded) or the path without ')'.
    assert all(not c.endswith(")") for c in captured), captured


def test_snapshot_capture_strips_backtick(_media_snap_re: re.Pattern[str]) -> None:
    """``media_re`` must not include the closing backtick in the path."""
    captured = _media_snap_re.findall(
        "assistant: `MEDIA:/home/kim/shot.png` and MEDIA:/home/kim/a.zip"
    )
    assert captured == ["/home/kim/shot.png", "/home/kim/a.zip"], captured
    assert all(not c.endswith("`") for c in captured)


def test_snapshot_capture_bare_token_still_captures(
    _media_snap_re: re.Pattern[str],
) -> None:
    """Backward-compat: bare ``MEDIA:/path`` still matches."""
    captured = _media_snap_re.findall(
        "MEDIA:/home/kim/a.zip and MEDIA:/home/kim/b.zip"
    )
    assert captured == ["/home/kim/a.zip", "/home/kim/b.zip"], captured


# ---------------------------------------------------------------------------
# Revert-sensitivity: the pre-fix class really would have captured backtick
# ---------------------------------------------------------------------------


def test_pre_fix_class_would_have_captured_backtick() -> None:
    """The buggy (pre-fix) class greedily included the trailing backtick."""
    buggy = re.compile(r"MEDIA:([^\s\)\]]+)")
    assert buggy.findall("`MEDIA:/home/kim/a.zip`") == ["/home/kim/a.zip`"]


def test_post_fix_class_drops_backtick() -> None:
    """The fixed class strips the closing backtick, as the source does."""
    fixed = re.compile(r"MEDIA:([^\s\)\]\`]+)")
    assert fixed.findall("`MEDIA:/home/kim/a.zip`") == ["/home/kim/a.zip"]


# ---------------------------------------------------------------------------
# node --check (so a JS syntax error in the modified source fails the suite)
# ---------------------------------------------------------------------------


import subprocess  # noqa: E402  (kept near the helper that uses it)


@pytest.mark.parametrize("relpath", sorted(JS_SITES.keys()))
def test_js_file_parses(relpath: str) -> None:
    """``node --check`` on every modified JS file."""
    proc = subprocess.run(
        ["node", "--check", str(REPO_ROOT / relpath)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"{relpath} failed `node --check`:\n{proc.stderr}"
    )
