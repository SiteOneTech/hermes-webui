"""Regression tests for WebUI session prefill parity."""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path


def test_prefill_script_output_becomes_safe_user_message(tmp_path):
    from api.streaming import _load_webui_prefill_context

    script = tmp_path / "recall.py"
    script.write_text("print('JOPLIN SESSION RECALL\\nCurrent Context: loaded')\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    result = _load_webui_prefill_context(
        {"prefill_messages_script": str(script)},
        python_exe=sys.executable,
        env={"PATH": os.environ.get("PATH", "")},
    )

    assert result["status"] == "loaded"
    assert result["source"] == "script"
    assert result["label"] == "recall.py"
    assert result["message_count"] == 1
    assert result["messages"] == [
        {
            "role": "user",
            "content": "JOPLIN SESSION RECALL\nCurrent Context: loaded",
        }
    ]


def test_prefill_script_uses_short_ttl_cache_keyed_by_path_and_mtime(tmp_path):
    from api.streaming import _load_webui_prefill_context

    script = tmp_path / "recall.py"
    counter = tmp_path / "counter.txt"
    script.write_text(
        "from pathlib import Path\n"
        f"p=Path({str(counter)!r})\n"
        "n=int(p.read_text() or '0') if p.exists() else 0\n"
        "p.write_text(str(n+1))\n"
        "print(f'run {n+1}')\n",
        encoding="utf-8",
    )

    first = _load_webui_prefill_context(
        {"prefill_messages_script": str(script)},
        python_exe=sys.executable,
        env={"PATH": os.environ.get("PATH", "")},
        script_cache_ttl=10.0,
    )
    second = _load_webui_prefill_context(
        {"prefill_messages_script": str(script)},
        python_exe=sys.executable,
        env={"PATH": os.environ.get("PATH", "")},
        script_cache_ttl=10.0,
    )

    assert first["messages"] == [{"role": "user", "content": "run 1"}]
    assert second["messages"] == [{"role": "user", "content": "run 1"}]
    assert counter.read_text(encoding="utf-8") == "1"

    script.write_text(script.read_text(encoding="utf-8") + "# invalidate\n", encoding="utf-8")
    third = _load_webui_prefill_context(
        {"prefill_messages_script": str(script)},
        python_exe=sys.executable,
        env={"PATH": os.environ.get("PATH", "")},
        script_cache_ttl=10.0,
    )

    assert third["messages"] == [{"role": "user", "content": "run 2"}]
    assert counter.read_text(encoding="utf-8") == "2"


def test_prefill_script_timeout_returns_error_without_hanging(tmp_path):
    from api.streaming import _load_webui_prefill_context

    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(1)\nprint('late')\n", encoding="utf-8")

    result = _load_webui_prefill_context(
        {"prefill_messages_script": str(script)},
        python_exe=sys.executable,
        env={"PATH": os.environ.get("PATH", "")},
        timeout=0.05,
        script_cache_ttl=0,
    )

    assert result["status"] == "error"
    assert result["source"] == "script"
    assert result["messages"] == []
    assert "timed out" in result["error"].lower()


def test_prefill_json_file_keeps_valid_roles_and_drops_invalid_items(tmp_path):
    from api.streaming import _load_webui_prefill_context

    prefill = tmp_path / "prefill.json"
    prefill.write_text(
        json.dumps(
            [
                {"role": "user", "content": "Pinned context"},
                {"role": "tool", "content": "drop invalid role"},
                {"role": "assistant", "content": "Useful assistant context"},
                {"role": "system", "content": "   "},
                "not a message",
            ]
        ),
        encoding="utf-8",
    )

    result = _load_webui_prefill_context({"prefill_messages_file": str(prefill)}, python_exe=sys.executable)

    assert result["status"] == "loaded"
    assert result["source"] == "file"
    assert result["label"] == "prefill.json"
    assert result["messages"] == [
        {"role": "user", "content": "Pinned context"},
        {"role": "assistant", "content": "Useful assistant context"},
    ]


def test_public_prefill_status_strips_message_bodies():
    from api.streaming import _public_prefill_context_status

    public = _public_prefill_context_status(
        {
            "status": "loaded",
            "source": "script",
            "label": "recall.py",
            "message_count": 1,
            "messages": [{"role": "user", "content": "private recall payload"}],
        }
    )

    assert public == {
        "status": "loaded",
        "source": "script",
        "label": "recall.py",
        "message_count": 1,
    }
    assert "messages" not in public


def test_prefill_context_redacts_secret_shaped_errors(tmp_path):
    from api.streaming import _load_webui_prefill_context

    script = tmp_path / "leaky.py"
    script.write_text("import sys\nsys.stderr.write('sk-proj-abcdefghijklmnopqrstuvwxyz123456 leaked')\nsys.exit(2)\n", encoding="utf-8")
    result = _load_webui_prefill_context(
        {"prefill_messages_script": str(script)},
        python_exe=sys.executable,
        env={"PATH": os.environ.get("PATH", "")},
        script_cache_ttl=0,
    )

    assert result["status"] == "error"
    assert result["messages"] == []
    assert "sk-proj" not in result.get("error", "")
    assert "[REDACTED]" in result.get("error", "")
