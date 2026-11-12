"""
Exposure-selection file (TOML) with optional user-defined groups (v1.16).

Hierarchy: MJD -> group (optional) -> exposure numbers. Exposures without a group belong to
DEFAULT_GROUP ("all"); in the outputs, the column ``group`` holds the group of each exposure.

    # one table per MJD; `expnums` without a group -> "all"
    [60216]
    expnums = [5736]

    # one group for the whole MJD
    [60225]
    group = "LV"
    expnums = ["6320-6328"]           # integers or "a-b" ranges (strings)

    # several groups in one MJD: sub-tables ...
    [60853.MCs]
    expnums = [42857, 42858]
    [60853."MW plane"]                # names with spaces in quotes
    expnums = [42859]

    # ... or the short form  <group> = [...]  inside the MJD table
    [60404]
    "MW plane" = [15610, 15613]
    DIG = [15611]

Rules: an exposure may appear only once (else ValueError); the MJD is checked against the drpall /
SFrame header (a mismatch is logged, the exposure is kept). Group names are free text; "expnums" and
"group" are reserved keys. Parsed with tomllib (Python >= 3.11) or tomli if available, otherwise with
the small built-in parser below (tables, strings, integers, booleans, arrays, comments).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)
DEFAULT_GROUP = "all"
RESERVED = ("expnums", "group")


# ----------------------------------------------------------------------------- TOML reading
def _load_toml(text: str) -> Dict[str, Any]:
    try:
        import tomllib  # Python >= 3.11
        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass
    try:
        import tomli
        return tomli.loads(text)
    except ModuleNotFoundError:
        return _mini_toml(text)


_KEY = r'(?:[A-Za-z0-9_-]+|"(?:[^"\\]|\\.)*"|\'[^\']*\')'


def _split_key(s: str, lineno: int) -> List[str]:
    parts, s = [], s.strip()
    while s:
        m = re.match(r"\s*(" + _KEY + r")\s*", s)
        if not m:
            raise ValueError(f"line {lineno}: bad key {s!r}")
        k = m.group(1)
        parts.append(_unquote(k) if k[0] in "\"'" else k)
        s = s[m.end():]
        if s:
            if s[0] != ".":
                raise ValueError(f"line {lineno}: bad key {s!r}")
            s = s[1:]
    return parts


def _unquote(s: str) -> str:
    if s.startswith('"'):
        return bytes(s[1:-1], "utf-8").decode("unicode_escape")
    return s[1:-1]


def _strip_comment(line: str) -> str:
    out, q = [], None
    for i, ch in enumerate(line):
        if q:
            if ch == q and not (q == '"' and i and line[i - 1] == "\\"):
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def _value(s: str, lineno: int):
    s = s.strip()
    if s.startswith("["):
        inner, items, q, cur, depth = s[1:-1], [], None, "", 0
        if not s.endswith("]"):
            raise ValueError(f"line {lineno}: unterminated array")
        for ch in inner:
            if q:
                cur += ch
                if ch == q:
                    q = None
            elif ch in "\"'":
                q = ch
                cur += ch
            elif ch == "[":
                depth += 1
                cur += ch
            elif ch == "]":
                depth -= 1
                cur += ch
            elif ch == "," and depth == 0:
                items.append(cur)
                cur = ""
            else:
                cur += ch
        items.append(cur)
        return [_value(x, lineno) for x in items if x.strip()]
    if s[:1] in "\"'":
        return _unquote(s)
    if s in ("true", "false"):
        return s == "true"
    try:
        return int(s.replace("_", ""))
    except ValueError:
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"line {lineno}: unsupported value {s!r}") from None


def _mini_toml(text: str) -> Dict[str, Any]:
    """Subset of TOML sufficient for exposure files (no inline tables, no multi-line strings)."""
    root: Dict[str, Any] = {}
    cur = root
    pending, start = "", 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        if pending:                       # continuation of a multi-line array
            pending += " " + line
            if pending.count("[") > pending.count("]"):
                continue
            line, pending, lineno = pending, "", start
        if not line:
            continue
        if line.startswith("[") and "=" not in line:
            if line.startswith("[["):
                raise ValueError(f"line {lineno}: arrays of tables are not supported")
            cur = root
            for k in _split_key(line[1:-1], lineno):
                cur = cur.setdefault(k, {})
                if not isinstance(cur, dict):
                    raise ValueError(f"line {lineno}: {k!r} is not a table")
            continue
        m = re.match(r"\s*(" + _KEY + r"(?:\s*\.\s*" + _KEY + r")*)\s*=\s*(.*)$", line)
        if not m:
            raise ValueError(f"line {lineno}: cannot parse {raw!r}")
        val = m.group(2)
        if val.startswith("[") and val.count("[") > val.count("]"):
            pending, start = line, lineno
            continue
        keys = _split_key(m.group(1), lineno)
        d = cur
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        if keys[-1] in d:
            raise ValueError(f"line {lineno}: duplicate key {keys[-1]!r}")
        d[keys[-1]] = _value(val, lineno)
    if pending:
        raise ValueError(f"line {start}: unterminated array")
    return root


# ----------------------------------------------------------------------------- exposure file
def _ints(vals, where: str) -> List[int]:
    if isinstance(vals, (int, str)):
        vals = [vals]
    if not isinstance(vals, list):
        raise ValueError(f"{where}: expected a list of exposure numbers, got {vals!r}")
    out: List[int] = []
    for v in vals:
        if isinstance(v, bool):
            raise ValueError(f"{where}: bad exposure number {v!r}")
        if isinstance(v, int):
            out.append(v)
            continue
        for part in str(v).replace(";", ",").replace(" ", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part[1:]:
                a, b = (int(x) for x in part.split("-", 1))
                out.extend(range(min(a, b), max(a, b) + 1))
            else:
                out.append(int(part))
    return out


def read_exposure_file(path) -> Dict[int, Tuple[int, str]]:
    """{expnum: (mjd, group)} from an exposure-selection TOML file (see module docstring)."""
    path = Path(path)
    data = _load_toml(path.read_text(encoding="utf-8"))
    out: Dict[int, Tuple[int, str]] = {}

    def add(mjd: int, group: str, vals, where: str) -> None:
        group = str(group).strip() or DEFAULT_GROUP
        if group.isdigit():
            log.warning("%s: group name %r looks like an MJD (in TOML, 'key = value' lines after a [table] "
                        "header belong to that table: put plain 'MJD = [...]' lines before the first table)",
                        where, group)
        for e in _ints(vals, where):
            if e in out and out[e] != (mjd, group):
                raise ValueError(f"{path.name}: expnum {e} listed twice ({out[e][1]!r} MJD {out[e][0]} and "
                                 f"{group!r} MJD {mjd})")
            out[e] = (mjd, group)

    for k, v in data.items():
        try:
            mjd = int(k)
        except ValueError:
            raise ValueError(f"{path.name}: top-level key {k!r} is not an MJD") from None
        where = f"{path.name} [{k}]"
        if not isinstance(v, dict):          # 60216 = [5736]
            add(mjd, DEFAULT_GROUP, v, where)
            continue
        unknown = set(v) - set(RESERVED)
        if "group" in v and not isinstance(v["group"], str):
            raise ValueError(f"{where}: group must be a string")
        if "expnums" in v:
            add(mjd, v.get("group", DEFAULT_GROUP), v["expnums"], where)
        elif "group" in v:
            raise ValueError(f"{where}: 'group' given without 'expnums'")
        for g in sorted(unknown):
            sub = v[g]
            if isinstance(sub, dict):        # [60853.MCs] expnums = [...]
                if set(sub) - {"expnums"}:
                    raise ValueError(f"{where}.{g}: only 'expnums' is allowed, got {sorted(sub)}")
                add(mjd, g, sub.get("expnums", []), f"{where}.{g}")
            else:                            # MCs = [...]
                add(mjd, g, sub, f"{where}.{g}")
    if not out:
        raise ValueError(f"{path.name}: no exposures")
    return out


def check_mjds(spec: Dict[int, Tuple[int, str]], rows) -> None:
    """Log exposures whose MJD in the file differs from the exposure table (they are kept)."""
    if rows is None or not len(rows) or "mjd" not in rows.colnames:
        return
    have = {int(e): int(m) for e, m in zip(rows["expnum"], rows["mjd"])}
    bad = [(e, mj, have[e]) for e, (mj, _) in sorted(spec.items()) if e in have and have[e] != mj]
    if bad:
        log.warning("%d exposure(s) with a different MJD in the exposure file than in the exposure table "
                    "(kept): %s", len(bad), ", ".join(f"{e}: file {a}, table {b}" for e, a, b in bad[:20]))


def write_exposure_file(path, spec: Dict[int, Tuple[int, str]], comments: Dict[int, str] = None,
                        header: str = "") -> Path:
    """Write {expnum: (mjd, group)} in the canonical layout (one table per MJD; group sub-tables if
    more than one group in the MJD); comments: optional {expnum: text} written next to each line."""
    by_mjd: Dict[int, Dict[str, List[int]]] = {}
    for e, (mjd, g) in spec.items():
        by_mjd.setdefault(mjd, {}).setdefault(g, []).append(e)
    lines = [f"# {x}" if x else "#" for x in header.splitlines()] + ([""] if header else [])

    def q(g):
        return g if re.fullmatch(r"[A-Za-z0-9_-]+", g) else '"' + g.replace('"', '\\"') + '"'

    def arr(es):
        if not comments:
            return "expnums = [" + ", ".join(str(e) for e in sorted(es)) + "]"
        return "expnums = [\n" + "".join(f"    {e},{'  # ' + comments[e] if comments.get(e) else ''}\n"
                                        for e in sorted(es)) + "]"

    for mjd in sorted(by_mjd):
        groups = by_mjd[mjd]
        if len(groups) == 1:
            g, es = next(iter(groups.items()))
            lines.append(f"[{mjd}]")
            if g != DEFAULT_GROUP:
                lines.append('group = "' + g.replace('"', '\\"') + '"')
            lines += [arr(es), ""]
        else:
            for g, es in sorted(groups.items()):
                lines += [f"[{mjd}.{q(g)}]", arr(es), ""]
    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
