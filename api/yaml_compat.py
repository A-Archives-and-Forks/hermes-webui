"""PyYAML-compatible ``safe_load``/``safe_dump``/``dump`` with a ruamel.yaml fallback.

Hermes Agent's managed dependency environment ships ruamel.yaml only, so WebUI code
imports YAML through this module instead of assuming PyYAML is installed.
"""

from __future__ import annotations

import io

try:
    import yaml as _pyyaml
except ImportError:
    _pyyaml = None
    import ruamel.yaml  # noqa: F401  (neither backend -> ImportError, as a bare ``import yaml`` would)

BACKEND = "pyyaml" if _pyyaml is not None else "ruamel"


def _ruamel(*, default_flow_style=False, allow_unicode=True, sort_keys=True, indent=None, width=None):
    from ruamel.yaml import YAML

    y = YAML(typ="safe", pure=True)
    y.default_flow_style = default_flow_style
    y.allow_unicode = allow_unicode
    y.representer.sort_base_mapping_type_on_output = sort_keys
    if indent is not None:
        y.indent = indent
    if width is not None:
        y.width = width
    return y


def safe_load(stream):
    if _pyyaml is not None:
        return _pyyaml.safe_load(stream)
    return _ruamel().load(stream)


def safe_dump(data, stream=None, *, default_flow_style=False, allow_unicode=False,
              sort_keys=True, indent=None, width=None):
    if _pyyaml is not None:
        return _pyyaml.safe_dump(
            data, stream, default_flow_style=default_flow_style, allow_unicode=allow_unicode,
            sort_keys=sort_keys, indent=indent, width=width,
        )
    y = _ruamel(default_flow_style=default_flow_style, allow_unicode=allow_unicode,
                sort_keys=sort_keys, indent=indent, width=width)
    if stream is not None:
        y.dump(data, stream)
        return None
    buf = io.StringIO()
    y.dump(data, buf)
    return buf.getvalue()


# WebUI only dumps plain dict/list/scalar config trees, which PyYAML's dump and
# safe_dump render identically.
dump = safe_dump
