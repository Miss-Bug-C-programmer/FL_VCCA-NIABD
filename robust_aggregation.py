from __future__ import annotations

from typing import Callable, Dict, Sequence

import torch

from numeric_integrity import require_finite_tensor


AGGREGATION_ALGORITHM_VERSION = "aggregation-v1-probability-space"


def _stack_probabilities(logits: Sequence[torch.Tensor], temperature: float) -> torch.Tensor:
    if not logits:
        raise ValueError("Robust aggregation requires at least one teacher.")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive.")
    shapes = {tuple(item.shape) for item in logits}
    if len(shapes) != 1:
        raise ValueError("All teacher logits must share one [P,C] shape.")
    if len(next(iter(shapes))) != 2:
        raise ValueError("Teacher logits must have shape [P,C].")
    stacked = torch.stack([
        torch.softmax(item.detach().float() / float(temperature), dim=1)
        for item in logits
    ], dim=0)
    require_finite_tensor(stacked, phase="aggregation", metric="teacher_probabilities")
    return stacked


def _validate_simplex(probabilities: torch.Tensor) -> torch.Tensor:
    result = probabilities.clamp_min(0.0)
    result = result / result.sum(dim=1, keepdim=True).clamp_min(torch.finfo(result.dtype).eps)
    require_finite_tensor(result, phase="aggregation", metric="aggregated_probabilities")
    if not torch.allclose(result.sum(dim=1), torch.ones(result.shape[0]), atol=1e-5, rtol=1e-5):
        raise ValueError("Aggregated probabilities are not on the simplex.")
    return result


def mean_probabilities(logits: Sequence[torch.Tensor], *, temperature: float) -> torch.Tensor:
    return _validate_simplex(_stack_probabilities(logits, temperature).mean(dim=0))


def coordinate_median_probabilities(logits: Sequence[torch.Tensor], *, temperature: float) -> torch.Tensor:
    return _validate_simplex(torch.median(_stack_probabilities(logits, temperature), dim=0).values)


def symmetric_trimmed_mean_probabilities(
    logits: Sequence[torch.Tensor],
    *,
    temperature: float,
    trim_fraction: float = 0.1,
) -> torch.Tensor:
    if not 0.0 <= float(trim_fraction) < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5).")
    probabilities = _stack_probabilities(logits, temperature)
    trim = int(float(trim_fraction) * probabilities.shape[0])
    if 2 * trim >= int(probabilities.shape[0]):
        raise ValueError("trim_fraction removes all teachers.")
    ordered = torch.sort(probabilities, dim=0).values
    return _validate_simplex(ordered[trim: probabilities.shape[0] - trim].mean(dim=0))


def confidence_consistency_filtered_mean(
    logits: Sequence[torch.Tensor],
    *,
    temperature: float,
    minimum_teachers: int = 2,
    distance_quantile: float = 0.75,
) -> torch.Tensor:
    if int(minimum_teachers) < 1:
        raise ValueError("minimum_teachers must be positive.")
    if not 0.0 < float(distance_quantile) <= 1.0:
        raise ValueError("distance_quantile must be in (0, 1].")
    probabilities = _stack_probabilities(logits, temperature)
    median = torch.median(probabilities, dim=0).values
    distance = (probabilities - median.unsqueeze(0)).abs().mean(dim=(1, 2))
    confidence = probabilities.max(dim=2).values.mean(dim=1)
    score = distance - 0.1 * confidence
    keep_count = max(int(minimum_teachers), int(torch.ceil(torch.tensor(float(probabilities.shape[0]) * float(distance_quantile))).item()))
    keep_count = min(keep_count, int(probabilities.shape[0]))
    keep = torch.topk(score, k=keep_count, largest=False).indices
    return _validate_simplex(probabilities[keep].mean(dim=0))


AGGREGATORS: Dict[str, Callable[..., torch.Tensor]] = {
    "mean-soft-probabilities": mean_probabilities,
    "mean-probabilities": mean_probabilities,
    "median-probabilities": coordinate_median_probabilities,
    "trimmed-mean-probabilities": symmetric_trimmed_mean_probabilities,
    "confidence-consistency-filtered-mean": confidence_consistency_filtered_mean,
}


def aggregate_probabilities(
    logits: Sequence[torch.Tensor],
    *,
    method: str = "mean-soft-probabilities",
    temperature: float = 2.0,
    trim_fraction: float = 0.1,
) -> torch.Tensor:
    key = str(method).lower()
    try:
        aggregator = AGGREGATORS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown probability-space aggregator: {method!r}") from exc
    kwargs = {"temperature": float(temperature)}
    if key == "trimmed-mean-probabilities":
        kwargs["trim_fraction"] = float(trim_fraction)
    return aggregator(logits, **kwargs)
