"""Console-noise regressions: git-info 404 on state.db-only sessions, pollers
that keep 409ing after a profile-cookie flip, and the unused pwa-startup preload."""
import io
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pytest

import api.routes as routes

REPO = Path(__file__).resolve().parent.parent
MESSAGES_JS = (REPO / "static" / "messages.js").read_text(encoding="utf-8")
SESSIONS_JS = (REPO / "static" / "sessions.js").read_text(encoding="utf-8")
INDEX_HTML = (REPO / "static" / "index.html").read_text(encoding="utf-8")


class _Handler:
    def __init__(self):
        self.wfile = io.BytesIO()
        self.headers = {}
        self.status = None

    def send_response(self, status):
        self.status = status

    def send_header(self, *_a):
        pass

    def end_headers(self):
        pass

    def payload(self):
        return json.loads(self.wfile.getvalue().decode("utf-8"))


def _git_info(sid):
    h = _Handler()
    routes.handle_get(h, urlparse(f"/api/git-info?session_id={sid}"))
    return h.status, h.payload()


@pytest.fixture
def git_repo(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    run("init", "-q", "-b", "main")
    (tmp_path / "a.txt").write_text("a")
    run("add", "a.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (tmp_path / "b.txt").write_text("b")
    return tmp_path


def _state_db_only(monkeypatch, sid, row):
    def _missing(*_a, **_kw):
        raise KeyError(sid)

    monkeypatch.setattr(routes, "get_session", _missing)
    monkeypatch.setattr(routes, "get_cli_sessions", lambda *_a, **_kw: [row] if row else [])


def test_git_info_serves_state_db_only_session(monkeypatch, git_repo):
    sid = "20260923_175914_e47592"
    _state_db_only(monkeypatch, sid, {"session_id": sid, "workspace": str(git_repo)})
    monkeypatch.setattr(routes, "resolve_trusted_workspace", lambda p, *_a, **_kw: Path(p))

    status, body = _git_info(sid)

    assert status == 200, body
    assert body["git"]["is_git"] is True
    assert body["git"]["branch"] == "main"
    assert body["git"]["untracked"] == 1


def test_git_info_untrusted_state_db_workspace_is_not_inspected(monkeypatch, git_repo):
    sid = "20260923_175914_e9d4bd"
    _state_db_only(monkeypatch, sid, {"session_id": sid, "workspace": str(git_repo)})

    def _untrusted(*_a, **_kw):
        raise ValueError("outside trusted roots")

    monkeypatch.setattr(routes, "resolve_trusted_workspace", _untrusted)
    assert _git_info(sid) == (200, {"git": None})


def test_git_info_state_db_session_without_workspace_has_no_badge(monkeypatch):
    sid = "20260923_000000_abcdef"
    _state_db_only(monkeypatch, sid, {"session_id": sid, "workspace": ""})
    monkeypatch.setattr(
        routes, "resolve_trusted_workspace",
        lambda *_a, **_kw: pytest.fail("must not fall back to the default workspace"),
    )
    assert _git_info(sid) == (200, {"git": None})


def test_git_info_unknown_session_still_404s(monkeypatch):
    _state_db_only(monkeypatch, "nope", None)
    status, body = _git_info("nope")
    assert status == 404 and body["error"] == "Session not found"


# ── frontend pollers, driven through node with the real functions ─────────────

def _extract_fn(src, name):
    start = src.index(f"function {name}(")
    brace = src.index("{", src.index(")", start))
    depth, i = 1, brace + 1
    while depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[start:i]


_HARNESS = r"""
var S = {session: {session_id: 'sid1'}, busy: false};
var calls = {api: 0, warn: 0, hide: 0};
var _approvalPollTimer = null, _approvalEventSource = null, _approvalSSEHealthTimer = null;
var _approvalFallbackPollInFlight = false, _approvalPollingSessionId = 'sid1';
var _clarifyEventSource = null, _clarifyFallbackTimer = null, _clarifyHealthTimer = null;
var _clarifyFallbackPollInFlight = false, _clarifyPollingSessionId = null, _clarifyMissingEndpointWarned = false;
var _approvalPendingBySession = new Map();
function _approvalPollingSessionMissingOrMismatched(sid){ return !sid || !S.session || S.session.session_id !== sid; }
function _hideApprovalCardIfOwner(){ calls.hide++; }
function _hideClarifyCardIfOwner(){ calls.hide++; }
function _clearApprovalPendingForSession(){}
function _clearClarifyPendingForSession(){}
function showApprovalForSession(){}
function showClarifyForSession(){}
function setComposerStatus(){}
function showToast(){}
console.warn = function(){ calls.warn++; };
var timers = [];
setInterval = function(fn){ timers.push(fn); return timers.length; };
clearInterval = function(){};
async function api(){
  calls.api++;
  var e = new Error('Session belongs to a different profile');
  e.status = STATUS;
  e.body = JSON.stringify(BODY);
  throw e;
}
"""


def _run_poller(start_fn, status, body):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    fns = [
        _extract_fn(SESSIONS_JS, "_sessionProfileMismatchFromError"),
        _extract_fn(MESSAGES_JS, "_startApprovalFallbackPoll"),
        _extract_fn(MESSAGES_JS, "stopApprovalPollingForSession"),
        _extract_fn(MESSAGES_JS, "stopApprovalPolling"),
        _extract_fn(MESSAGES_JS, "_startClarifyFallbackPoll"),
        _extract_fn(MESSAGES_JS, "stopClarifyPollingForSession"),
        _extract_fn(MESSAGES_JS, "stopClarifyPolling"),
    ]
    script = (
        _HARNESS.replace("STATUS", str(status)).replace("BODY", json.dumps(body))
        + "\n".join(fns)
        + f"""
(async () => {{
  {start_fn}('sid1');
  await new Promise(r => setTimeout(r, 0));
  const pollingAfterFirst = {'_approvalPollingSessionId' if 'Approval' in start_fn else '_clarifyPollingSessionId'};
  process.stdout.write(JSON.stringify({{calls, pollingAfterFirst}}));
}})();
"""
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


_MISMATCH = {"error": "Session belongs to a different profile", "code": "session_profile_mismatch",
             "session_id": "sid1", "profile": "deepseek"}


def test_approval_poll_stops_on_profile_mismatch_409():
    r = _run_poller("_startApprovalFallbackPoll", 409, _MISMATCH)
    assert r["calls"]["api"] == 1
    assert r["pollingAfterFirst"] is None, "approval poller must stop after a session_profile_mismatch 409"
    assert r["calls"]["hide"] == 0, "a live approval card must stay standing"


def test_approval_poll_keeps_running_on_other_errors():
    r = _run_poller("_startApprovalFallbackPoll", 500, {"error": "boom"})
    assert r["pollingAfterFirst"] == "sid1"


def test_clarify_poll_stops_quietly_on_profile_mismatch_409():
    r = _run_poller("_startClarifyFallbackPoll", 409, _MISMATCH)
    assert r["calls"]["api"] == 1
    assert r["pollingAfterFirst"] is None, "clarify poller must stop after a session_profile_mismatch 409"
    assert r["calls"]["warn"] == 0, "expected profile mismatch must not log '[clarify] pending poll failed'"
    assert r["calls"]["hide"] == 0, "a live clarify card must stay standing"


def test_clarify_poll_still_warns_on_unexpected_errors():
    r = _run_poller("_startClarifyFallbackPoll", 500, {"error": "boom"})
    assert r["calls"]["warn"] == 1
    assert r["pollingAfterFirst"] == "sid1"


def test_pwa_startup_is_not_preloaded_next_to_its_own_blocking_script():
    assert '<script src="static/pwa-startup.js?v=__WEBUI_VERSION__"></script>' in INDEX_HTML
    assert 'rel="preload" href="static/pwa-startup.js' not in INDEX_HTML
