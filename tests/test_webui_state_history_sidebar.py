"""Regression coverage for WebUI-origin state.db history in the sidebar."""
from __future__ import annotations

import json
from urllib.parse import urlparse


class _FakeHandler:
    """Minimal BaseHTTPRequestHandler stand-in for routes.handle_get."""

    def __init__(self):
        self.status = None
        self.sent_headers = []
        self.body = bytearray()
        self.wfile = self

    def send_response(self, code):
        self.status = code

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        pass

    def write(self, data):
        self.body.extend(data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8"))

    def get_json(self):
        return json.loads(self.body.decode("utf-8"))


def test_api_sessions_includes_webui_state_db_history_when_cli_sessions_hidden(monkeypatch):
    """WebUI history mirrored into state.db must not depend on the CLI toggle.

    `show_cli_sessions` controls external CLI/agent rows. WebUI-origin rows are
    first-party chat history, so they must remain visible even when external
    sessions are hidden.
    """
    from api import profiles, routes

    monkeypatch.setattr(
        routes,
        "all_sessions",
        lambda diag=None: [
            {
                "session_id": "current-webui-sidecar",
                "title": "Current WebUI chat",
                "source_tag": "webui",
                "raw_source": "webui",
                "session_source": "webui",
                "profile": "default",
                "message_count": 2,
                "updated_at": 200.0,
            }
        ],
    )
    monkeypatch.setattr(
        routes,
        "get_state_webui_sessions",
        lambda: [
            {
                "session_id": "historical-webui-state-row",
                "title": "Historical WebUI chat",
                "source_tag": "webui",
                "raw_source": "webui",
                "session_source": "webui",
                "profile": "default",
                "message_count": 4,
                "updated_at": 150.0,
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        routes,
        "get_cli_sessions",
        lambda: [
            {
                "session_id": "external-cli-row",
                "title": "External CLI chat",
                "source_tag": "cli",
                "raw_source": "cli",
                "session_source": "cli",
                "profile": "default",
                "message_count": 5,
                "updated_at": 175.0,
                "is_cli_session": True,
            }
        ],
    )
    monkeypatch.setattr(
        routes,
        "load_settings",
        lambda: {
            "show_cli_sessions": False,
            "show_previous_messaging_sessions": False,
        },
    )
    monkeypatch.setattr(routes, "_reconcile_stale_stream_state_for_session_rows", lambda rows: False)
    monkeypatch.setattr(routes, "_session_attention_summary", lambda sid: None)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")

    handler = _FakeHandler()
    routes.handle_get(handler, urlparse("/api/sessions"))
    assert handler.status == 200
    payload = handler.get_json()
    ids = {row["session_id"] for row in payload["sessions"]}

    assert "current-webui-sidecar" in ids
    assert "historical-webui-state-row" in ids
    assert "external-cli-row" not in ids
