"""Nested subagent rows drop the agent's "Subagent: " title prefix."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = (Path(__file__).resolve().parent.parent / "static" / "sessions.js").read_text()


def _fn(name):
    m = re.search(r"^function " + name + r"\(.*?^}\n", SRC, re.S | re.M)
    assert m, name
    return m.group(0)


@pytest.mark.skipif(not shutil.which("node"), reason="node required")
def test_subagent_prefix_stripped_only_for_delegated_children():
    rows = [
        {"title": "Subagent: STRUCT-GO: implement parsers", "parent_session_id": "p",
         "relationship_type": "child_session", "raw_source": "subagent"},
        {"title": "Subagent: plain chat", "raw_source": "webui"},
        {"title": "Subagent:", "parent_session_id": "p",
         "relationship_type": "child_session", "raw_source": "subagent"},
    ]
    js = "\n".join(_fn(n) for n in ("_isChildSession", "_isDelegatedSubagentRow", "_sessionDisplayTitle"))
    js += f"\nconsole.log(JSON.stringify({json.dumps(rows)}.map(_sessionDisplayTitle)));"
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == ["STRUCT-GO: implement parsers", "Subagent: plain chat", "Untitled"]
