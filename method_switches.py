from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_METHOD_FLAGS = {
    "baseline": (False, False),
    "vcaa": (True, False),
    "niabd": (False, True),
    "vcaa-niabd": (True, True),
}


@dataclass(frozen=True)
class MethodSwitches:
    """Canonical experiment mechanism switches.

    VCAA owns teacher admission. NIABD owns post-admission purification.
    Keeping the two booleans explicit prevents a defense-only run from
    accidentally acquiring admission authority through orchestration code.
    """

    method: str
    enable_vcaa: bool
    enable_niabd: bool

    @property
    def admission_enabled(self) -> bool:
        return bool(self.enable_vcaa)

    @property
    def defense_enabled(self) -> bool:
        return bool(self.enable_niabd)


def strategy_name(enable_vcaa: bool, enable_niabd: bool) -> str:
    flags = (bool(enable_vcaa), bool(enable_niabd))
    for method, expected in _METHOD_FLAGS.items():
        if flags == expected:
            return method
    raise RuntimeError(f"Unreachable mechanism switch combination: {flags!r}")


def resolve_method_switches(
    method: Optional[str],
    *,
    enable_vcaa: bool = False,
    enable_niabd: bool = False,
) -> MethodSwitches:
    """Resolve CLI alias/legacy flags to one unambiguous mechanism state.

    If neither ``--method`` nor an enable flag is supplied, the result is the
    baseline: VCAA disabled and NIABD disabled.  A supplied ``--method`` is
    authoritative; legacy enable flags may accompany it only when they encode
    exactly the same state.
    """

    requested = str(method or "").strip().lower()
    explicit_flags = (bool(enable_vcaa), bool(enable_niabd))

    if requested:
        if requested not in _METHOD_FLAGS:
            raise ValueError(
                f"Unsupported method={requested!r}; expected one of "
                f"{sorted(_METHOD_FLAGS)}."
            )
        method_flags = _METHOD_FLAGS[requested]
        if any(explicit_flags) and explicit_flags != method_flags:
            raise ValueError(
                "--method conflicts with explicit --enable-vcaa/"
                "--enable-niabd flags."
            )
        resolved_flags = method_flags
    else:
        resolved_flags = explicit_flags

    resolved_method = strategy_name(*resolved_flags)
    return MethodSwitches(
        method=resolved_method,
        enable_vcaa=resolved_flags[0],
        enable_niabd=resolved_flags[1],
    )
