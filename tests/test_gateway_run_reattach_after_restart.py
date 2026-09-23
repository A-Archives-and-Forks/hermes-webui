"""A gateway-owned run must survive a WebUI restart instead of being marked interrupted."""
from collections import OrderedDict
import io
import json
import threading
import urllib.error
from email.message import Message

import pytest

import api.gateway_chat as gateway_chat
import api.models as models
import api.streaming as streaming
from api.config import ACTIVE_RUNS, STREAMS, STREAMS_LOCK, create_stream_channel
from api.models import new_session


@pytest.fixture
def isolated_sessions(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", OrderedDict())
    monkeypatch.setenv("HERMES_WEBUI_CHAT_BACKEND", "gateway")
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_USE_RUNS_API", "1")
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://gateway.local")
    monkeypatch.setattr(gateway_chat, "GATEWAY_REATTACH_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(gateway_chat, "_gateway_reasoning_effort_for_request", lambda *a, **k: None)
    monkeypatch.setattr(streaming, "_load_webui_prefill_context", lambda cfg: {
        "status": "not_configured", "source": "none", "label": "", "message_count": 0, "messages": [],
    })
    monkeypatch.setattr(streaming, "_prefill_messages_with_webui_context", lambda ctx, cfg: [])
    return session_dir


def _orphaned_gateway_turn(run_id="run_survivor", stream_id="stream-before-restart", started_at=1.0):
    """Persist the sidecar exactly as a WebUI process leaves it when killed mid-run."""
    s = new_session()
    s.messages = [
        {"role": "user", "content": "earlier", "timestamp": 0.5},
        {"role": "assistant", "content": "earlier reply", "timestamp": 0.6},
    ]
    s.active_stream_id = stream_id
    s.pending_user_message = "long task"
    s.pending_attachments = []
    s.pending_started_at = started_at
    s.pending_user_source = "webui"
    s.gateway_run = {"run_id": run_id, "stream_id": stream_id, "regeneration": False, "goal_related": False}
    s.save()
    # Simulate the new process: nothing of the old run is in memory.
    models.SESSIONS.clear()
    with STREAMS_LOCK:
        STREAMS.pop(stream_id, None)
    ACTIVE_RUNS.pop(stream_id, None)
    return s.session_id, stream_id


def _wait_for_reattach_threads(timeout=10.0):
    for thread in threading.enumerate():
        if thread.name.startswith("gateway-reattach-"):
            thread.join(timeout)
            assert not thread.is_alive(), "reattach worker did not settle"


def test_runs_api_start_sends_idempotency_key_and_persists_run_id(isolated_sessions, monkeypatch):
    s = new_session()
    stream_id = "stream-live"
    s.active_stream_id = stream_id
    s.pending_user_message = "hi"
    s.pending_attachments = []
    s.pending_started_at = 123.0
    s.save()
    captured = {}

    class _Resp:
        def __init__(self, body=None, lines=None):
            self._body, self._lines = body, lines or []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *_a):
            return self._body

        def __iter__(self):
            return iter(self._lines)

    def fake_urlopen(req, timeout=None):
        if req.get_method() == "POST":
            captured["post_headers"] = dict(req.header_items())
            return _Resp(body=b'{"run_id":"run_live"}')
        # The run id is durable before the first event is relayed.
        captured["persisted_at_events"] = json.loads(
            (isolated_sessions / f"{s.session_id}.json").read_text()
        ).get("gateway_run")
        return _Resp(lines=[
            b'data: {"event":"message.delta","delta":"done"}\n',
            b'data: {"event":"run.completed","output":"done"}\n',
            b"data: [DONE]\n",
        ])

    monkeypatch.setattr(gateway_chat, "gateway_supports_approval", lambda *a, **k: True)
    monkeypatch.setattr(gateway_chat.urllib.request, "urlopen", fake_urlopen)
    with STREAMS_LOCK:
        STREAMS[stream_id] = create_stream_channel()

    gateway_chat._run_gateway_chat_streaming(s.session_id, "hi", "test-model", "/tmp", stream_id, [])

    assert captured["post_headers"]["Idempotency-key"] == f"webui-{stream_id}"
    assert captured["persisted_at_events"]["run_id"] == "run_live"
    assert captured["persisted_at_events"]["stream_id"] == stream_id
    assert "api_key" not in captured["persisted_at_events"]
    saved = json.loads((isolated_sessions / f"{s.session_id}.json").read_text())
    assert saved["gateway_run"] is None
    assert saved["active_stream_id"] is None
    assert saved["messages"][-1]["content"] == "done"


def test_restart_reattaches_and_writes_back_real_answer(isolated_sessions, monkeypatch):
    sid, stream_id = _orphaned_gateway_turn()
    release = threading.Event()
    polls = []

    def fake_status(base_url, api_key, run_id):
        polls.append(run_id)
        if not release.is_set():
            return {"run_id": run_id, "status": "running"}
        return {
            "run_id": run_id,
            "status": "completed",
            "output": "finished after the restart",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }

    monkeypatch.setattr(gateway_chat, "_get_gateway_run_status", fake_status)

    assert gateway_chat.resume_gateway_runs_after_restart() == [sid]
    # While the run is still going, the turn is live: no stale-pending repair, reconnect works.
    assert stream_id in STREAMS
    assert gateway_chat.wait_for_gateway_run_id(stream_id, 5.0) == (True, "run_survivor")
    live = models.get_session(sid)
    assert live.active_stream_id == stream_id
    assert not any(m.get("_error") for m in live.messages)

    release.set()
    _wait_for_reattach_threads()

    saved = json.loads((isolated_sessions / f"{sid}.json").read_text())
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant", "user", "assistant"]
    assert saved["messages"][2]["content"] == "long task"
    assert saved["messages"][3]["content"] == "finished after the restart"
    assert not any(m.get("_error") for m in saved["messages"])
    assert saved["active_stream_id"] is None
    assert saved["pending_user_message"] is None
    assert saved["gateway_run"] is None
    assert set(polls) == {"run_survivor"}
    assert stream_id not in STREAMS


def test_reattach_reports_run_the_gateway_no_longer_knows(isolated_sessions, monkeypatch):
    sid, _stream_id = _orphaned_gateway_turn(run_id="run_gone")

    def fake_status(base_url, api_key, run_id):
        raise urllib.error.HTTPError("http://gateway.local", 404, "not found", Message(), io.BytesIO(b"{}"))

    monkeypatch.setattr(gateway_chat, "_get_gateway_run_status", fake_status)
    assert gateway_chat.resume_gateway_runs_after_restart() == [sid]
    _wait_for_reattach_threads()

    saved = json.loads((isolated_sessions / f"{sid}.json").read_text())
    assert saved["messages"][-2]["content"] == "long task"
    assert saved["messages"][-1]["_error"] is True
    assert "could not be recovered after the WebUI restart" in saved["messages"][-1]["content"]
    assert saved["active_stream_id"] is None
    assert saved["gateway_run"] is None


def test_reattach_surfaces_pending_approval_once(isolated_sessions, monkeypatch):
    sid, stream_id = _orphaned_gateway_turn(run_id="run_parked")
    relayed = []
    calls = {"n": 0}

    def fake_status(base_url, api_key, run_id):
        calls["n"] += 1
        if calls["n"] < 4:
            return {"run_id": run_id, "status": "waiting_for_approval", "approval": {
                "event": "approval.request", "approval_id": "appr-1", "command": "rm -rf build", "description": "delete",
            }}
        return {"run_id": run_id, "status": "completed", "output": "approved and done"}

    monkeypatch.setattr(gateway_chat, "_get_gateway_run_status", fake_status)
    monkeypatch.setattr(
        gateway_chat, "_relay_gateway_run_approval",
        lambda session_id, run_id, payload, *a, **k: relayed.append((session_id, run_id, payload["approval_id"])),
    )
    gateway_chat.resume_gateway_runs_after_restart()
    _wait_for_reattach_threads()

    assert relayed == [(sid, "run_parked", "appr-1")]
    saved = json.loads((isolated_sessions / f"{sid}.json").read_text())
    assert saved["messages"][-1]["content"] == "approved and done"


def test_reattach_is_default_off_without_gateway_backend(isolated_sessions, monkeypatch):
    _orphaned_gateway_turn()
    monkeypatch.delenv("HERMES_WEBUI_CHAT_BACKEND")
    monkeypatch.setattr(
        gateway_chat, "_get_gateway_run_status",
        lambda *a, **k: pytest.fail("legacy backend must not poll the gateway"),
    )
    assert gateway_chat.resume_gateway_runs_after_restart() == []


def test_sidecar_without_gateway_run_is_left_to_stale_pending_repair(isolated_sessions, monkeypatch):
    sid, stream_id = _orphaned_gateway_turn()
    s = models.Session.load(sid)
    assert s is not None
    s.gateway_run = None
    s.save()
    monkeypatch.setattr(
        gateway_chat, "_get_gateway_run_status",
        lambda *a, **k: pytest.fail("no recorded run id: nothing to reattach"),
    )
    assert gateway_chat.resume_gateway_runs_after_restart() == []
    assert stream_id not in STREAMS


def test_cancel_still_stops_a_reattached_run(isolated_sessions, monkeypatch):
    sid, stream_id = _orphaned_gateway_turn(run_id="run_to_stop")
    stopped = []
    monkeypatch.setattr(
        gateway_chat, "_get_gateway_run_status",
        lambda base_url, api_key, run_id: {"run_id": run_id, "status": "running"},
    )
    monkeypatch.setattr(gateway_chat, "stop_gateway_run", lambda run_id: stopped.append(run_id) or True)
    gateway_chat.resume_gateway_runs_after_restart()

    # Same two steps /api/chat/cancel performs for a gateway-backed stream.
    _structured, run_id = gateway_chat.wait_for_gateway_run_id(stream_id, 5.0)
    assert gateway_chat.stop_gateway_run(run_id)
    for _ in range(500):
        if stream_id in gateway_chat.CANCEL_FLAGS:
            break
        threading.Event().wait(0.01)
    assert streaming.cancel_stream(stream_id)
    _wait_for_reattach_threads()

    assert stopped == ["run_to_stop"]
    saved = json.loads((isolated_sessions / f"{sid}.json").read_text())
    assert saved["active_stream_id"] is None
    assert saved["gateway_run"] is None
    assert not any(m.get("content") == "long task" and m.get("role") == "assistant" for m in saved["messages"])


def _poll_until_completed(monkeypatch, seen):
    def fake_status(base_url, api_key, run_id):
        seen.append((base_url, api_key, run_id))
        return {"run_id": run_id, "status": "completed", "output": "answer"}

    monkeypatch.setattr(gateway_chat, "_get_gateway_run_status", fake_status)


@pytest.mark.parametrize("index_state", ["missing", "stale", "corrupt"])
def test_reattach_does_not_depend_on_the_session_index(isolated_sessions, monkeypatch, index_state):
    sid, _stream_id = _orphaned_gateway_turn()
    index = isolated_sessions / "_index.json"
    if index_state == "missing":
        index.unlink(missing_ok=True)
    elif index_state == "stale":
        # Sidecar saved, index update lost: the row still shows no active stream.
        index.write_text(json.dumps([{"session_id": sid, "active_stream_id": None}]))
    else:
        index.write_text("{not json")
    seen = []
    _poll_until_completed(monkeypatch, seen)

    assert gateway_chat.resume_gateway_runs_after_restart() == [sid]
    _wait_for_reattach_threads()

    saved = json.loads((isolated_sessions / f"{sid}.json").read_text())
    assert saved["messages"][-1]["content"] == "answer"
    assert saved["active_stream_id"] is None


def test_reattach_skips_idle_sidecars_without_parsing_them(isolated_sessions, monkeypatch):
    idle = new_session()
    idle.messages = [{"role": "user", "content": "x", "timestamp": 1.0}]
    idle.save()
    sid, _ = _orphaned_gateway_turn()
    loaded = []
    real_load = models.Session.load_metadata_only
    monkeypatch.setattr(
        models.Session, "load_metadata_only",
        classmethod(lambda cls, s, **kw: loaded.append(s) or real_load.__func__(cls, s, **kw)),
    )
    _poll_until_completed(monkeypatch, [])

    assert gateway_chat.resume_gateway_runs_after_restart() == [sid]
    _wait_for_reattach_threads()
    assert idle.session_id not in loaded
    assert sid in loaded


def test_reattach_resolves_gateway_from_the_session_profile(isolated_sessions, monkeypatch):
    """The process may restart under another default profile; poll the session profile's gateway."""
    import contextlib
    import os
    from api import profiles

    sid, stream_id = _orphaned_gateway_turn()
    s = models.Session.load(sid)
    s.profile = "work"
    s.save(touch_updated_at=False)
    models.SESSIONS.clear()
    scopes = []

    @contextlib.contextmanager
    def fake_scope(profile_name, purpose="", logger_override=None):
        scopes.append(profile_name)
        env = {"work": ("http://work-gateway:8642", "work-key")}.get(profile_name)
        old = {k: os.environ.get(k) for k in ("HERMES_WEBUI_GATEWAY_BASE_URL", "HERMES_WEBUI_GATEWAY_API_KEY")}
        if env:
            os.environ["HERMES_WEBUI_GATEWAY_BASE_URL"], os.environ["HERMES_WEBUI_GATEWAY_API_KEY"] = env
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    monkeypatch.setattr(profiles, "profile_scope_for_detached_worker", fake_scope)
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_API_KEY", "default-key")
    seen = []
    _poll_until_completed(monkeypatch, seen)

    assert gateway_chat.resume_gateway_runs_after_restart() == [sid]
    _wait_for_reattach_threads()

    assert scopes == ["work"]
    assert {(b, k) for b, k, _ in seen} == {("http://work-gateway:8642", "work-key")}
    saved = json.loads((isolated_sessions / f"{sid}.json").read_text())
    assert saved["messages"][-1]["content"] == "answer"


@pytest.mark.parametrize("code", [401, 403])
def test_reattach_auth_rejection_fails_fast_with_an_auth_error(isolated_sessions, monkeypatch, code):
    sid, _stream_id = _orphaned_gateway_turn()
    calls = []

    def rejected(base_url, api_key, run_id):
        calls.append(run_id)
        raise urllib.error.HTTPError("http://gateway.local/v1/runs/x", code, "denied", Message(), io.BytesIO(b""))

    monkeypatch.setattr(gateway_chat, "_get_gateway_run_status", rejected)
    gateway_chat.resume_gateway_runs_after_restart()
    _wait_for_reattach_threads()

    assert len(calls) == 1
    saved = json.loads((isolated_sessions / f"{sid}.json").read_text())
    assert saved["active_stream_id"] is None and saved["gateway_run"] is None
    assert f"HTTP {code}" in json.dumps(saved["messages"][-1])
