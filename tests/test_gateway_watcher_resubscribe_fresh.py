"""Regression: a re-subscribing SSE client gets a fresh watcher projection.

``_poll_once`` resets the parity timestamp when it runs with no subscribers
("force fresh projection on the next subscription"), but since the idle-park
change (#7694) the poll loop parks *before* calling ``_poll_once`` once the
last subscriber leaves, so that reset is not reached on a normal disconnect.
A tab that reconnects inside the parity interval then resumes with the stale
cheap fingerprint and skips the projection. The reset now happens in
``unsubscribe()`` when the last subscriber leaves.
"""
from __future__ import annotations

import importlib
import sqlite3
import time
from pathlib import Path


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            session_source TEXT,
            model TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            parent_session_id TEXT,
            message_count INTEGER DEFAULT 0,
            title TEXT,
            archived INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, source, model, started_at, message_count, title) "
        "VALUES ('tg1', 'telegram', 'm', ?, 1, 'Chat')",
        (time.time(),),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('tg1', 'user', 'x', ?)",
        (time.time(),),
    )
    conn.commit()
    conn.close()
    return db


def test_resubscribe_after_last_unsubscribe_forces_fresh_projection(tmp_path):
    gw = importlib.import_module("api.gateway_watcher")
    watcher = gw.GatewayWatcher(state_db_path=_make_db(tmp_path))
    q = watcher.subscribe()
    assert watcher._poll_once(now=1.0) is True

    watcher.unsubscribe(q)
    assert watcher._last_cheap_fp == ""
    assert watcher._last_full_projection_at is None

    watcher.subscribe()
    # Well inside PROJECTION_PARITY_INTERVAL and with an unchanged state.db.
    assert watcher._poll_once(now=2.0) is True


def test_unsubscribe_keeps_fingerprint_while_other_subscribers_remain(tmp_path):
    gw = importlib.import_module("api.gateway_watcher")
    watcher = gw.GatewayWatcher(state_db_path=_make_db(tmp_path))
    first = watcher.subscribe()
    watcher.subscribe()
    assert watcher._poll_once(now=1.0) is True
    fingerprint = watcher._last_cheap_fp
    assert fingerprint

    watcher.unsubscribe(first)
    assert watcher._last_cheap_fp == fingerprint
    assert watcher._last_full_projection_at == 1.0
    assert watcher._poll_once(now=2.0) is False


def test_unsubscribe_of_unknown_queue_does_not_reset(tmp_path):
    import queue as _queue

    gw = importlib.import_module("api.gateway_watcher")
    watcher = gw.GatewayWatcher(state_db_path=_make_db(tmp_path))
    watcher.subscribe()
    assert watcher._poll_once(now=1.0) is True
    fingerprint = watcher._last_cheap_fp

    watcher.unsubscribe(_queue.Queue())
    assert watcher._last_cheap_fp == fingerprint
    assert watcher._last_full_projection_at == 1.0


def test_poll_loop_reprojects_for_reconnecting_subscriber(tmp_path, monkeypatch):
    """Through the real loop: disconnect, park, reconnect -> projection runs again."""
    gw = importlib.import_module("api.gateway_watcher")
    db = _make_db(tmp_path)
    projections: list[float] = []
    real_projection = gw._get_agent_sessions_from_db

    def tracing_projection(path):
        projections.append(time.monotonic())
        return real_projection(path)

    monkeypatch.setattr(gw, "_get_agent_sessions_from_db", tracing_projection)
    watcher = gw.GatewayWatcher(state_db_path=db)
    watcher.POLL_INTERVAL = 0.05
    watcher.start()
    try:
        q = watcher.subscribe()
        deadline = time.monotonic() + 2.0
        while not projections and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(projections) == 1

        watcher.unsubscribe(q)
        time.sleep(watcher.POLL_INTERVAL * 4)  # loop parks; no further projection
        assert len(projections) == 1

        watcher.subscribe()
        deadline = time.monotonic() + 2.0
        while len(projections) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(projections) == 2, "reconnecting subscriber must get a fresh projection"
    finally:
        watcher.stop()
