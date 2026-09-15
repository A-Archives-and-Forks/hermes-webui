"""Regression: unread state must not be clobbered by a second WebUI client.

Two WebUI windows/tabs on the same origin and browser profile share one
``localStorage``. Each client caches the unread stores
(``hermes-session-viewed-counts`` and ``hermes-session-completion-unread``) in
module state and serialises that cache wholesale on save. With no ``storage``
listener for either key, a stale client overwrites the other client's writes:

- a session's *viewed* count can be rolled backwards, so ``message_count >
  viewed_count`` re-lights a dot the user just cleared; and
- a completion-unread marker cleared in one client is resurrected by the other.

This fixes the class by (a) max-merging viewed counts on save so a count can
never go down or drop another client's entries, and (b) invalidating both
caches on ``storage`` events so a concurrent write is re-read instead of
overwritten.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_JS = (ROOT / "static" / "sessions.js").read_text(encoding="utf-8")


def _extract(name: str) -> str:
    """Extract a top-level `function name(...) { ... }` definition by brace match."""
    marker = f"function {name}("
    start = SESSIONS_JS.index(marker)
    brace = SESSIONS_JS.index("{", start)
    depth = 0
    for i in range(brace, len(SESSIONS_JS)):
        ch = SESSIONS_JS[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return SESSIONS_JS[start:i + 1]
    raise AssertionError(f"could not brace-match {name}")


def _run_node(script: str) -> dict:
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


_VIEWED_FNS = [
    "_getSessionViewedCounts",
    "_saveSessionViewedCounts",
    "_setSessionViewedCount",
    "_clearSessionViewedCount",
]
_UNREAD_FNS = [
    "_getSessionCompletionUnread",
    "_saveSessionCompletionUnread",
    "_markSessionCompletionUnread",
    "_clearSessionCompletionUnread",
    "_hasSessionCompletionUnread",
]

_HEADER = """
const _store = {};
const localStorage = {
  getItem: (k) => (k in _store ? _store[k] : null),
  setItem: (k, v) => { _store[k] = String(v); },
};
const SESSION_VIEWED_COUNTS_KEY = 'hermes-session-viewed-counts';
const SESSION_COMPLETION_UNREAD_KEY = 'hermes-session-completion-unread';
"""


def _make_client_block():
    """A factory that defines one client's module state and returns its API."""
    fns = "\n".join(_extract(n) for n in _VIEWED_FNS + _UNREAD_FNS)
    handler = ""
    if "function _handleUnreadStorageEvent(" in SESSIONS_JS:
        handler = _extract("_handleUnreadStorageEvent")
    return f"""
function makeClient() {{
  let _sessionViewedCounts = null;
  let _sessionCompletionUnread = null;
  {fns}
  {handler}
  return {{
    setViewed: _setSessionViewedCount,
    clearViewed: _clearSessionViewedCount,
    getViewed: (sid) => {{
      const c = _getSessionViewedCounts();
      return c[sid] === undefined ? null : c[sid];
    }},
    markUnread: _markSessionCompletionUnread,
    clearUnread: _clearSessionCompletionUnread,
    hasUnread: _hasSessionCompletionUnread,
    handleStorage: (typeof _handleUnreadStorageEvent === 'function') ? _handleUnreadStorageEvent : null,
  }};
}}
"""


def test_viewed_count_is_monotonic_across_clients():
    """A stale client must not roll a session's viewed count backwards.

    Fails before the fix: client B serialises its stale cache and overwrites
    client A's newer value, so the cleared dot re-lights via the count rule.
    """
    script = _HEADER + _make_client_block() + """
const A = makeClient();
const B = makeClient();
A.setViewed('X', 417);
B.setViewed('X', 131);          // stale client believes the session has 131 msgs
console.log(JSON.stringify({
  disk: _store['hermes-session-viewed-counts'] ? JSON.parse(_store['hermes-session-viewed-counts']).X : null,
  bCache: B.getViewed('X'),
}));
"""
    out = _run_node(script)
    assert out["disk"] == 417, (
        "a stale client must not lower a session's viewed count (disk was "
        f"{out['disk']}, expected 417)"
    )
    assert out["bCache"] == 417, (
        "the stale client's cache must be re-synced to the merged count"
    )


def test_clear_viewed_count_still_removes_from_disk():
    """Guard: session deletion must still prune the viewed-count entry."""
    script = _HEADER + _make_client_block() + """
const A = makeClient();
A.setViewed('D', 5);
A.clearViewed('D');
const parsed = JSON.parse(_store['hermes-session-viewed-counts'] || '{}');
console.log(JSON.stringify({ diskStillHas: Object.prototype.hasOwnProperty.call(parsed, 'D') }));
"""
    out = _run_node(script)
    assert out["diskStillHas"] is False, "clearing a viewed count must remove it from disk"


def test_storage_event_invalidates_unread_caches():
    """A storage event for either unread key must drop the in-memory cache so a
    concurrent client's write is re-read, not clobbered.

    Skipped before the fix (the handler does not exist yet); passes after.
    """
    if "function _handleUnreadStorageEvent(" not in SESSIONS_JS:
        import pytest
        pytest.skip("_handleUnreadStorageEvent not present yet")

    script = _HEADER + _make_client_block() + """
const A = makeClient();
const B = makeClient();
A.markUnread('Y', 3);            // A records a completion
B.hasUnread('Y');                // B loads the marker into its cache
A.clearUnread('Y');              // A clears it (e.g. the user visited Y in A)
// B sees A's storage event and must forget its cached copy.
B.handleStorage({ key: 'hermes-session-completion-unread' });
B.markUnread('Z', 7);            // B writes for an unrelated session
const disk = JSON.parse(_store['hermes-session-completion-unread'] || '{}');
console.log(JSON.stringify({
  yResurrected: Object.prototype.hasOwnProperty.call(disk, 'Y'),
  zPresent: Object.prototype.hasOwnProperty.call(disk, 'Z'),
}));
"""
    out = _run_node(script)
    assert out["zPresent"] is True, "B's new marker must be persisted"
    assert out["yResurrected"] is False, (
        "a marker cleared in one client must not be resurrected by another"
    )


def test_unread_storage_handler_is_wired():
    """The window storage listener must route to the unread-cache handler."""
    assert "void _handleUnreadStorageEvent(e);" in SESSIONS_JS, (
        "the storage listener must invoke _handleUnreadStorageEvent"
    )
