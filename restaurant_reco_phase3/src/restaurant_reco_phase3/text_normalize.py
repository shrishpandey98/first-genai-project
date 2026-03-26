from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")


def norm(s: object) -> str:
    if s is None:
        return ""
    out = str(s).strip().lower()
    out = _WS_RE.sub(" ", out)
    return out


def token_set(s: object) -> set[str]:
    n = norm(s)
    if not n:
        return set()
    return set(n.split(" "))

