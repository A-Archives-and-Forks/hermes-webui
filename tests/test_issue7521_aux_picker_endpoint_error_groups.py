"""Auxiliary-task pickers must keep provider groups that only carry an endpoint error (#7521).

``GET /api/models`` keeps a named custom provider group with ``models: []`` plus
a ``models_endpoint_error`` payload when its ``/v1/models`` probe fails (that
contract is pinned by ``tests/test_issue2540_models_endpoint_error.py`` and is
rendered by the main picker in ``static/ui.js``). The auxiliary-task pickers in
``static/panels.js`` dropped every zero-model group, so a provider with an
unreachable models endpoint silently disappeared from both the provider and the
model selects — the user could neither see the provider nor the reason it had no
models.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PANELS_JS_PATH = ROOT / "static" / "panels.js"
PANELS_JS = PANELS_JS_PATH.read_text(encoding="utf-8")
NODE = shutil.which("node")

# Shared node prelude: read panels.js, pull a top-level ``function name(...)``
# out of the bundle by brace matching, and evaluate it in isolation (same
# technique as tests/test_auxiliary_models_settings.py).
NODE_PRELUDE = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[1], 'utf8');

function extract(name){
  const re = new RegExp('function\\s+' + name + '\\s*\\(');
  const start = src.search(re);
  if(start < 0) throw new Error(name + ' not found');
  let i = src.indexOf('{', start);
  let depth = 0;
  while(i < src.length){
    const ch = src[i];
    if(ch === '{') depth += 1;
    else if(ch === '}') {
      depth -= 1;
      if(depth === 0){
        break;
      }
    }
    i += 1;
  }
  if(depth !== 0) throw new Error(name + ' parse failed');
  return src.slice(start, i + 1);
}

function _stubElement(tag){
  return {
    tagName: String(tag || 'option').toUpperCase(),
    children: [],
    dataset: {},
    disabled: false,
    selected: false,
    _value: '',
    _text: '',
    _html: '',
    set value(v){ this._value = v; },
    get value(){ return this._value; },
    set textContent(v){ this._text = v; },
    get textContent(){ return this._text; },
    set innerHTML(v){ this._html = v; if(v === '') this.children = []; },
    get innerHTML(){ return this._html; },
    appendChild(child){ this.children.push(child); return child; },
    insertBefore(child, ref){
      const idx = this.children.indexOf(ref);
      if(idx < 0){ this.children.push(child); } else { this.children.splice(idx, 0, child); }
      return child;
    },
    get options(){ return this.children; },
  };
}

global.document = { createElement: (tag) => _stubElement(tag) };
global.t = (key) => key;
"""


def _run_node(script: str) -> dict:
    proc = subprocess.run(
        [NODE, "-e", script, str(PANELS_JS_PATH)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"node probe failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_aux_load_builds_providers_from_shared_helper():
    """panels.js must build the auxiliary provider list in one named helper."""
    assert "function _auxProvidersFromModelGroups" in PANELS_JS, (
        "Missing _auxProvidersFromModelGroups() in panels.js"
    )
    assert "_auxProviders=_auxProvidersFromModelGroups(groups)" in PANELS_JS, (
        "Auxiliary load flow must build the provider list through the helper"
    )
    legacy_inline_filter = "_auxProviders=groups.filter(g=>g.provider&&((g.models&&g.models.length>0)"
    assert legacy_inline_filter not in PANELS_JS, (
        "The zero-model group filter must live in _auxProvidersFromModelGroups, not inline"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_endpoint_error_group_survives_the_zero_model_filter():
    """A group with models_endpoint_error must be kept; empty groups without it must not."""
    script = (
        NODE_PRELUDE
        + r"""
eval(extract('_auxProvidersFromModelGroups'));

const groups = [
  {provider:'openai-codex', provider_id:'openai-codex', models:[{id:'gpt-5.5', label:'GPT-5.5'}]},
  {provider:'Broken Proxy', provider_id:'custom:broken-proxy', models:[],
   models_endpoint_error:{kind:'network', code:null, message:'Models endpoint unreachable for broken-proxy; verify base_url.'}},
  {provider:'Empty Extra', provider_id:'custom:empty-extra', models:[], extra_models:[]},
  null,
];

const providers = _auxProvidersFromModelGroups(groups);
console.log(JSON.stringify({
  slugs: providers.map((p) => p.slug),
  names: providers.map((p) => p.name),
  brokenError: (providers.find((p) => p.slug === 'custom:broken-proxy') || {}).modelsEndpointError || null,
  brokenModels: (providers.find((p) => p.slug === 'custom:broken-proxy') || {}).models || null,
  healthyModels: ((providers.find((p) => p.slug === 'openai-codex') || {}).models || []).length,
}));
"""
    )
    result = _run_node(script)

    assert result["slugs"] == ["openai-codex", "custom:broken-proxy"], (
        "Endpoint-error group must survive the filter while plain empty groups are dropped"
    )
    assert result["names"] == ["openai-codex", "Broken Proxy"]
    assert result["brokenError"] == {
        "kind": "network",
        "code": None,
        "message": "Models endpoint unreachable for broken-proxy; verify base_url.",
    }, "models_endpoint_error must be carried into the auxiliary provider entry"
    assert result["brokenModels"] == []
    assert result["healthyModels"] == 1


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_model_select_surfaces_provider_endpoint_error():
    """The model select must explain the empty list instead of looking model-less."""
    script = (
        NODE_PRELUDE
        + r"""
eval(extract('_modelBareNameForProvider'));
eval(extract('_buildAuxModelOptions'));

const message = 'Models endpoint returned 502 for broken-proxy; see logs.';
const broken = [{slug:'custom:broken-proxy', name:'Broken Proxy', models:[],
                 modelsEndpointError:{kind:'http', code:502, message}}];
const withModels = [{slug:'custom:allowed-proxy', name:'Allowed Proxy',
                     models:[{id:'allowed/one', label:'allowed/one'}],
                     modelsEndpointError:{kind:'network', code:null, message:'Models endpoint unreachable for allowed-proxy; verify base_url.'}}];

const brokenSel = document.createElement('select');
const brokenCanonical = _buildAuxModelOptions(brokenSel, 'custom:broken-proxy', broken, 'broken/manual');

const allowedSel = document.createElement('select');
_buildAuxModelOptions(allowedSel, 'custom:allowed-proxy', withModels, '');

console.log(JSON.stringify({
  canonical: brokenCanonical,
  firstValue: brokenSel.children[0] ? brokenSel.children[0].value : null,
  firstDisabled: brokenSel.children[0] ? !!(brokenSel.children[0].disabled) : null,
  hints: brokenSel.children.filter((o) => o.dataset && o.dataset.modelsEndpointError === '1')
    .map((o) => ({value:o.value, disabled:!!(o.disabled), text:o.textContent})),
  customPresent: brokenSel.children.some((o) => o.value === '__custom__'),
  configuredPreserved: brokenSel.children.some((o) => o.value === 'broken/manual' && o.selected === true),
  allowedHints: allowedSel.children.filter((o) => o.dataset && o.dataset.modelsEndpointError === '1').length,
  allowedModels: allowedSel.children.map((o) => o.value),
}));
"""
    )
    result = _run_node(script)

    assert result["canonical"] == "broken/manual", "configured model must stay canonical for the broken provider"
    assert result["firstValue"] == "", "the auto placeholder must stay the first option"
    assert result["firstDisabled"] is False, "the auto placeholder must remain selectable"
    assert result["hints"] == [{"value": "", "disabled": True, "text": "\u26a0 " + "Models endpoint returned 502 for broken-proxy; see logs."}], (
        "the model select must carry a disabled hint with the provider endpoint error message"
    )
    assert result["customPresent"] is True, "the custom-model escape hatch must stay available"
    assert result["configuredPreserved"] is True, "the configured model must stay selected"
    assert result["allowedHints"] == 1, "a provider with models must still surface its endpoint error"
    assert result["allowedModels"] == ["", "", "allowed/one", "__custom__"], (
        "models must still be listed after the auto placeholder and the error hint"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_model_select_without_endpoint_error_keeps_previous_shape():
    """Providers without an endpoint error must not gain a hint option."""
    script = (
        NODE_PRELUDE
        + r"""
eval(extract('_modelBareNameForProvider'));
eval(extract('_buildAuxModelOptions'));

const providers = [{slug:'openai-codex', name:'openai-codex', models:[{id:'gpt-5.5', label:'GPT-5.5'}]}];
const sel = document.createElement('select');
const canonical = _buildAuxModelOptions(sel, 'openai-codex', providers, 'gpt-5.5');

console.log(JSON.stringify({
  canonical,
  values: sel.children.map((o) => o.value),
  hints: sel.children.filter((o) => o.dataset && o.dataset.modelsEndpointError === '1').length,
  selected: sel.children.filter((o) => o.selected === true).map((o) => o.value),
}));
"""
    )
    result = _run_node(script)

    assert result["hints"] == 0
    assert result["values"] == ["", "gpt-5.5", "__custom__"]
    assert result["selected"] == ["gpt-5.5"]
