"""Regression tests for bounded sidebar session-list caching."""

from api import route_session_list_cache as cache
from api.models import Session


def _key():
    return cache._session_list_cache_key(
        active_profile="default",
        all_profiles=False,
        show_cli_sessions=False,
        show_previous_messaging_sessions=False,
        show_cron_sessions=False,
    )


def test_cache_drops_full_transcript_fields_before_storing(monkeypatch):
    cache._session_list_cache_clear()
    payload = {
        "sessions": [{
            "session_id": "huge",
            "title": "Large session",
            "message_count": 1,
            "messages": [{"role": "assistant", "content": "x" * 1000000}],
            "tool_calls": [{"arguments": "y" * 1000000}],
            "context_messages": ["z" * 1000000],
            "gateway_routing_history": [{"provider": "old"}] * 1000,
        }],
        "sidebar_reference_sessions": [],
        "settings": {"show_cli_sessions": False},
    }
    monkeypatch.setattr(cache, "_session_list_cache_resolved_source_stamp", lambda _key: ("stable",))

    cache._session_list_cache_set(_key(), payload)
    cached, fresh = cache._session_list_cache_get(_key(), allow_stale=False)

    assert fresh is True
    assert cached["sessions"] == [{
        "session_id": "huge",
        "title": "Large session",
        "message_count": 1,
    }]
    assert "messages" not in cached["sessions"][0]
    assert "tool_calls" not in cached["sessions"][0]
    assert "context_messages" not in cached["sessions"][0]
    assert "gateway_routing_history" not in cached["sessions"][0]


def test_cache_read_returns_independent_bounded_rows(monkeypatch):
    cache._session_list_cache_clear()
    monkeypatch.setattr(cache, "_session_list_cache_resolved_source_stamp", lambda _key: ("stable",))
    key = _key()
    cache._session_list_cache_set(key, {"sessions": [{"session_id": "s1", "title": "One"}]})

    first, _ = cache._session_list_cache_get(key)
    first["sessions"][0]["title"] = "mutated"
    second, _ = cache._session_list_cache_get(key)

    assert second["sessions"][0]["title"] == "One"


def test_sidebar_compact_omits_heavy_session_metadata():
    session = Session(
        session_id="heavy",
        title="Heavy",
        messages=[{"role": "user", "content": "hello"}],
        compression_anchor_summary="s" * 1000000,
        compression_anchor_details={"detail": "d" * 1000000},
        context_engine_state={"state": "x" * 1000000},
        compression_recovery={"recovery": "r" * 1000000},
        gateway_routing_history=[{"provider": "old"}] * 10000,
        composer_draft={"draft": "c" * 1000000},
        process_wakeup_pause={"pause": "p" * 1000000},
        share_token="token-value",
    )

    normal = session.compact()
    sidebar = session.compact(sidebar_metadata_only=True)

    assert normal["gateway_routing_history"]
    assert normal["compression_anchor_summary"]
    for field in (
        "compression_anchor_summary", "compression_anchor_details",
        "context_engine_state", "compression_recovery",
        "gateway_routing_history", "composer_draft",
        "process_wakeup_pause", "share_token",
    ):
        assert field not in sidebar
