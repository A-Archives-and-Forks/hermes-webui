"""PyYAML-compatible ``safe_load``/``safe_dump``/``dump`` with a ruamel.yaml fallback.

Hermes Agent's managed dependency environment ships ruamel.yaml only, so WebUI code
imports YAML through this module instead of assuming PyYAML is installed.

The ruamel fallback keeps PyYAML's YAML 1.1 semantics. ruamel defaults to YAML 1.2,
where only ``true``/``false`` are booleans: it would load ``on``/``off``/``yes``/``no``
as strings, and dump the strings ``"off"``/``"no"`` unquoted, which PyYAML and the
Agent's own reader (``hermes_yaml``, also pinned to 1.1) then read back as booleans.
``tool_progress: off`` is a real Hermes setting, so a save through a 1.2 dumper would
silently change it.
"""

from __future__ import annotations

import io

try:
    import yaml as _pyyaml
except ImportError:
    _pyyaml = None
    import ruamel.yaml  # noqa: F401  (neither backend -> ImportError, as a bare ``import yaml`` would)

BACKEND = "pyyaml" if _pyyaml is not None else "ruamel"

_YAML11 = (1, 1)
_resolver_cls = None


def _yaml11_resolver():
    """A resolver that applies YAML 1.1 implicit typing without writing a %YAML directive."""
    global _resolver_cls
    if _resolver_cls is None:
        from ruamel.yaml.resolver import VersionedResolver

        class _Yaml11Resolver(VersionedResolver):
            @property
            def processing_version(self):
                return _YAML11

        _resolver_cls = _Yaml11Resolver
    return _resolver_cls


def _ruamel(*, default_flow_style=False, allow_unicode=True, sort_keys=True, indent=None, width=None):
    from ruamel.yaml import YAML

    # A fresh instance per call: ruamel YAML objects are not thread-safe.
    y = YAML(typ="safe", pure=True)
    y.Resolver = _yaml11_resolver()
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
    from ruamel.yaml import YAML

    y = YAML(typ="safe", pure=True)
    y.version = _YAML11
    return y.load(stream)


def _ruamel_dump(data, stream, **options):
    y = _ruamel(**options)
    if stream is not None:
        y.dump(data, stream)
        return None
    buf = io.StringIO()
    y.dump(data, buf)
    return buf.getvalue()


def safe_dump(data, stream=None, *, default_flow_style=False, allow_unicode=False,
              sort_keys=True, indent=None, width=None):
    if _pyyaml is not None:
        return _pyyaml.safe_dump(
            data, stream, default_flow_style=default_flow_style, allow_unicode=allow_unicode,
            sort_keys=sort_keys, indent=indent, width=width,
        )
    return _ruamel_dump(data, stream, default_flow_style=default_flow_style,
                        allow_unicode=allow_unicode, sort_keys=sort_keys, indent=indent, width=width)


def dump(data, stream=None, *, default_flow_style=False, allow_unicode=False,
         sort_keys=True, indent=None, width=None):
    """``yaml.dump`` on the PyYAML backend, so existing call sites are byte-identical.

    WebUI only dumps plain dict/list/scalar config trees, so the ruamel fallback uses
    the same safe representer as ``safe_dump``.
    """
    if _pyyaml is not None:
        return _pyyaml.dump(
            data, stream, default_flow_style=default_flow_style, allow_unicode=allow_unicode,
            sort_keys=sort_keys, indent=indent, width=width,
        )
    return _ruamel_dump(data, stream, default_flow_style=default_flow_style,
                        allow_unicode=allow_unicode, sort_keys=sort_keys, indent=indent, width=width)
