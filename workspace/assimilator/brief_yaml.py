"""Writing a brief so a downstream formatter cannot silently corrupt it.

The shared commit hook runs yamlfmt over every staged file under ~/repos with
`retain_line_breaks: true`, and that setting replaces a line break INSIDE a
string value with the literal sentinel `#magic___^_^___line`. Where the value
also runs long it truncates the scalar and drops the closing quote, leaving
YAML that will not parse and a tail that is gone for good - not recoverable
from the brief, only from a regenerate.

The protection is not the quoting style, it is being on ONE PHYSICAL LINE. A
single-quoted scalar broken across two lines is corrupted exactly like a folded
one. So a value carrying a newline is written double-quoted with the break
escaped, and the emitter width is raised so it cannot wrap.

That width costs readability, and briefs are read by hand - so it is paid PER
FILE and only where it buys something. 23 of 748 briefs carry a newline-bearing
string; the other 725 keep the ordinary 80-column form.

The root fix is `retain_line_breaks: false` in the hook config, which is not
ours. This is defence in depth for a corruption that is both silent and lossy.
"""

from __future__ import annotations

import yaml

try:  # pragma: no cover - depends on the libyaml build
    from yaml import CSafeDumper as _BaseDumper
except ImportError:  # pragma: no cover
    from yaml import SafeDumper as _BaseDumper

_UNWRAPPED = 10**9


class _EscapingDumper(_BaseDumper):
    """Emits any string containing a newline as a one-line double-quoted scalar."""


def _represent_str(dumper, data):
    style = '"' if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_EscapingDumper.add_representer(str, _represent_str)


def contains_newline(obj) -> bool:
    if isinstance(obj, str):
        return "\n" in obj
    if isinstance(obj, dict):
        return any(contains_newline(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(contains_newline(v) for v in obj)
    return False


def dump(obj) -> str:
    """YAML for a brief, hardened only when the brief needs it."""
    if contains_newline(obj):
        return yaml.dump(
            obj,
            Dumper=_EscapingDumper,
            sort_keys=False,
            allow_unicode=True,
            width=_UNWRAPPED,
        )
    return yaml.dump(obj, Dumper=_BaseDumper, sort_keys=False, allow_unicode=True)
