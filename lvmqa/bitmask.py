"""
2-bit QA flag encoding and bitmask packing.

Each metric owns a fixed *slot*; its 2-bit code occupies bits (2*slot, 2*slot+1),
LSB-first (same convention as the DR20 qa_bitmask).

    GOOD = 00, WARNING = 01, MAJOR ISSUE = 10, UNDEFINED = 11
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

MAX_SLOTS = 31  # keeps the packed value inside a signed int64 (FITS 'K')


class QAFlag(IntEnum):
    GOOD = 0
    WARNING = 1
    MAJOR = 2
    UNDEFINED = 3

    @property
    def label(self) -> str:
        return FLAG_LABELS[self]


FLAG_LABELS: Dict[QAFlag, str] = {
    QAFlag.GOOD: "GOOD",
    QAFlag.WARNING: "WARNING",
    QAFlag.MAJOR: "MAJOR ISSUE",
    QAFlag.UNDEFINED: "UNDEFINED",
}
LABEL_TO_FLAG = {v: k for k, v in FLAG_LABELS.items()}

# severity used when combining flags: MAJOR > WARNING > UNDEFINED > GOOD
_SEVERITY = {QAFlag.GOOD: 0, QAFlag.UNDEFINED: 1, QAFlag.WARNING: 2, QAFlag.MAJOR: 3}


def combine(flags: Iterable[int]) -> QAFlag:
    """Worst flag of a set (MAJOR > WARNING > UNDEFINED > GOOD); empty -> UNDEFINED."""
    flags = [QAFlag(int(f)) for f in flags]
    if not flags:
        return QAFlag.UNDEFINED
    return max(flags, key=lambda f: _SEVERITY[f])


def pack(flags_by_slot: Mapping[int, int]) -> int:
    value = 0
    for slot, code in flags_by_slot.items():
        if not 0 <= int(slot) < MAX_SLOTS:
            raise ValueError(f"slot {slot} out of range [0, {MAX_SLOTS})")
        value |= (int(code) & 0b11) << (2 * int(slot))
    return value


def parse_bitmask(value: Any) -> int:
    """Accept int, numpy int, decimal string or '0x..' hex string."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, int):
        return value
    s = str(value).strip()
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def unpack(bitmask: Any, slot: int) -> QAFlag:
    return QAFlag((parse_bitmask(bitmask) >> (2 * int(slot))) & 0b11)


def decode_bitmask(bitmask: Any, slots: Mapping[str, int] | Sequence[Tuple[str, int]]
                   ) -> List[Tuple[str, str]]:
    """
    Decode a packed bitmask into [(metric_name, label), ...].

    ``slots`` maps metric name -> slot (e.g. from the DESC extension columns
    ``Metric`` and ``Slot``).
    """
    items = slots.items() if isinstance(slots, Mapping) else slots
    items = [(n, s) for n, s in items if s is not None and int(s) >= 0]  # skip non-bitmask metrics
    return [(name, unpack(bitmask, slot).label) for name, slot in sorted(items, key=lambda x: int(x[1]))]
