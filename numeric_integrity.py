import math
from typing import Any, Dict, Optional

import torch


_CONTEXT_KEYS = ("phase", "round", "hop", "sender", "receiver", "key", "metric", "value")


def _context_value(context: Optional[Dict[str, Any]], key: str, default: str = "unknown") -> Any:
    if isinstance(context, dict) and key in context:
        return context[key]
    return default


def format_integrity_context(
    *,
    phase: str,
    metric: str,
    value: Any,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    ctx = dict(context or {})
    ctx["phase"] = phase
    ctx["metric"] = metric
    ctx["value"] = value
    return " ".join(f"{key}={_context_value(ctx, key)}" for key in _CONTEXT_KEYS)


class NumericIntegrityError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: str = "unknown",
        metric: str = "unknown",
        value: Any = "unknown",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"{message} [{format_integrity_context(phase=phase, metric=metric, value=value, context=context)}]")


class SnapshotProfileError(NumericIntegrityError, ValueError):
    pass


class AccountingIntegrityError(NumericIntegrityError, ValueError):
    pass


def is_finite_value(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def require_finite_tensor(
    tensor: torch.Tensor,
    *,
    phase: str,
    metric: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    if not torch.is_tensor(tensor):
        raise NumericIntegrityError(
            "Expected tensor for finite check.",
            phase=phase,
            metric=metric,
            value=type(tensor).__name__,
            context=context,
        )
    if not bool(torch.isfinite(tensor).all().item()):
        bad = tensor.detach()
        value = "nonfinite"
        if bad.numel() > 0:
            flat = bad.flatten()
            mask = ~torch.isfinite(flat)
            if bool(mask.any().item()):
                value = float(flat[mask][0].detach().cpu().item())
        raise NumericIntegrityError(
            "Non-finite tensor encountered.",
            phase=phase,
            metric=metric,
            value=value,
            context=context,
        )


def require_finite_scalar(
    value: Any,
    *,
    phase: str,
    metric: str,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    try:
        out = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    except Exception as exc:
        raise NumericIntegrityError(
            "Non-numeric scalar encountered.",
            phase=phase,
            metric=metric,
            value=repr(value),
            context=context,
        ) from exc
    if not math.isfinite(out):
        raise NumericIntegrityError(
            "Non-finite scalar encountered.",
            phase=phase,
            metric=metric,
            value=out,
            context=context,
        )
    return out
