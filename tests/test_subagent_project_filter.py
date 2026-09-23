"""Delegated subagent children must follow their parent's project filter.

Subagent child sessions are created by the agent (state.db) and never carry a
``project_id`` of their own. The sidebar project filter compared each row's own
``project_id``, so selecting a project dropped every subagent child and the
parent rendered with no stacked children, while the Unassigned view kept them.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
SESSIONS_JS = (REPO_ROOT / "static" / "sessions.js").read_text(encoding="utf-8")
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")

_PREAMBLE = """
const src = %s;
function extractFunc(name) {
  const start = src.search(new RegExp('function\\\\s+' + name + '\\\\s*\\\\('));
  if (start < 0) throw new Error(name + ' not found');
  let i = src.indexOf('{', start) + 1, depth = 1;
  while (depth > 0 && i < src.length) { if (src[i] === '{') depth++; else if (src[i] === '}') depth--; i++; }
  return src.slice(start, i);
}
function _isCliSession(s){ return !!(s && (s.is_cli_session || s.session_source==='cli')); }
function _isExternalSession(s){ return !!(s && (s.is_cli_session || s.session_source === 'messaging')); }
function _isMessagingSession(s){ return !!(s && s.session_source==='messaging'); }
function _hasUnreadForSession(s){ return !!(s && s.has_unread); }
function _sessionDisplayTitle(s){ return s && s.title; }
Object.assign(global, {_isCliSession, _isExternalSession, _isMessagingSession, _hasUnreadForSession, _sessionDisplayTitle});
global.INFLIGHT = {};
global.NO_PROJECT_FILTER = '__no_project__';
global.window = {};
global._archivedCliCount = 0; global._archivedWebuiCount = 0;
global._sidebarReferenceSessions = [];
global.S = { session: null, busy: false, activeStreamId: null };
global._showArchived = false;
global._sessionSourceFilter = 'webui';
for (const fn of ['_isSessionLocallyStreaming','_hasPendingUserMessageSignal','_isSessionEffectivelyStreaming',
  '_isChildSession','_isForkWithResolvableParent','_sessionLineageKey','_sidebarLineageKeyForRow',
  '_collapseSessionLineageForSidebar','_attachChildSessionsToSidebarRows','_sessionAttentionState',
  '_sidebarRowHasVisibleMessages','_sidebarProjectIdForRow','_sidebarRowsById','_sidebarHasUnprojectedRows',
  '_partitionSidebarSessionRows','_scopedSidebarReferenceRows','_renderSidebarRowsFromRawSessions']) {
  eval.call(global, extractFunc(fn));
}
const parent = { session_id:'proj_parent', title:'Parent', session_source:'webui', raw_source:'webui', source_tag:'webui', message_count:5, project_id:'projX', updated_at:100, last_message_at:100 };
const child = { session_id:'sub_child', title:'Subagent Session', parent_session_id:'proj_parent', relationship_type:'child_session', raw_source:'subagent', source_tag:'subagent', session_source:'other', _parent_lineage_root_id:'proj_parent', _cross_surface_child_session:true, message_count:3, updated_at:101, last_message_at:101 };
const grandchild = { session_id:'sub_grandchild', title:'Nested subagent', parent_session_id:'sub_child', relationship_type:'child_session', raw_source:'subagent', source_tag:'subagent', session_source:'other', message_count:2, updated_at:102, last_message_at:102 };
function render(project, rows) {
  global._activeProject = project;
  const part = _partitionSidebarSessionRows(rows, null);
  const out = _renderSidebarRowsFromRawSessions(part.sessionsRaw, part.webuiReferenceRaw);
  const p = out.find(r => r.session_id === 'proj_parent') || {};
  return { raw: part.sessionsRaw.map(s => s.session_id).sort(), top: out.map(r => r.session_id),
           children: (p._child_sessions || []).map(c => c.session_id) };
}
"""


def _run(scenario: str) -> dict:
    assert NODE is not None
    source = (_PREAMBLE % json.dumps(SESSIONS_JS)) + scenario
    result = subprocess.run([NODE], input=source, cwd=str(REPO_ROOT),
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout.strip())


def test_subagent_child_is_shown_inside_parent_project():
    out = _run("console.log(JSON.stringify(render('projX', [parent, child])));")
    assert out["raw"] == ["proj_parent", "sub_child"]
    assert out["top"] == ["proj_parent"]
    assert out["children"] == ["sub_child"]


def test_nested_subagent_inherits_project_through_child():
    out = _run("console.log(JSON.stringify(render('projX', [parent, child, grandchild])));")
    assert out["raw"] == ["proj_parent", "sub_child", "sub_grandchild"]
    assert out["top"] == ["proj_parent"]


def test_subagent_of_project_parent_is_not_unassigned():
    out = _run("console.log(JSON.stringify(render(NO_PROJECT_FILTER, [parent, child])));")
    assert out["raw"] == []
    assert out["top"] == []


def test_subagent_excluded_from_other_project_and_own_project_id_wins():
    out = _run("""
const other = render('projY', [parent, child]);
const explicit = render('projY', [parent, Object.assign({}, child, {project_id:'projY'})]);
console.log(JSON.stringify({other, explicit}));
""")
    assert out["other"]["raw"] == []
    assert out["explicit"]["raw"] == ["sub_child"]


def test_parent_resolved_from_reference_rows():
    """Parent only present as a hidden reference row still scopes the child."""
    out = _run("""
global._sidebarReferenceSessions = [parent];
const rows = render('projX', [child]);
global._activeProject = 'projX';
const scoped = _scopedSidebarReferenceRows(false).map(s => s.session_id);
console.log(JSON.stringify({raw: rows.raw, scoped}));
""")
    assert out["raw"] == ["sub_child"]
    assert out["scoped"] == ["proj_parent"]


def test_unassigned_chip_ignores_subagents_of_project_parents():
    out = _run("console.log(JSON.stringify(_sidebarHasUnprojectedRows([parent, child], _sidebarRowsById([[parent, child]]))));")
    assert out is False


def test_unassigned_chip_uses_reference_parent():
    """A child whose project parent is only a reference row is not unassigned."""
    out = _run("""
global._sidebarReferenceSessions = [parent];
global._activeProject = null;
const part = _partitionSidebarSessionRows([child], null);
console.log(JSON.stringify({has: _sidebarHasUnprojectedRows(part.profileFiltered, part.rowsById),
  unassigned: render(NO_PROJECT_FILTER, [child]).raw}));
""")
    assert out["unassigned"] == []
    assert out["has"] is False
