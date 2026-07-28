from __future__ import annotations

import torch


def _require_image_batch(images: torch.Tensor) -> None:
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("Backdoor trigger expects images with shape [N,C,H,W].")
    if int(images.shape[-2]) < 4 or int(images.shape[-1]) < 4:
        raise ValueError("Images are too small for the configured trigger.")


def _patch_bounds(
    height: int,
    width: int,
    size: int,
    location: str,
    *,
    margin: int = 1,
) -> tuple[slice, slice]:
    max_size = max(1, min(height - 2 * margin, width - 2 * margin))
    size = max(1, min(int(size), max_size))
    if location == "bottom-right":
        y0, x0 = height - margin - size, width - margin - size
    elif location == "top-left":
        y0, x0 = margin, margin
    elif location == "top-right":
        y0, x0 = margin, width - margin - size
    elif location == "bottom-left":
        y0, x0 = height - margin - size, margin
    elif location == "center-edge":
        y0 = max(margin, height // 2 - size // 2)
        x0 = width - margin - size
    else:
        raise ValueError(f"Unknown trigger location={location!r}.")
    return slice(y0, y0 + size), slice(x0, x0 + size)


def apply_badnets(
    images: torch.Tensor,
    *,
    size: int,
    value: float = 1.0,
    location: str = "bottom-right",
) -> torch.Tensor:
    """Apply a fixed visible square patch in normalized image space."""

    _require_image_batch(images)
    output = images.clone()
    _, _, height, width = output.shape
    ys, xs = _patch_bounds(height, width, size, location)
    output[:, :, ys, xs] = float(value)
    return output


def _dba_origins(
    height: int,
    width: int,
    local_size: int,
) -> list[tuple[int, int]]:
    """Four separated sub-triggers whose union is the DBA global trigger."""

    gap = max(1, int(local_size))
    total = 2 * int(local_size) + gap
    top = max(1, height - total - 2)
    left = max(1, width - total - 2)
    return [
        (top, left),
        (top, left + int(local_size) + gap),
        (top + int(local_size) + gap, left),
        (
            top + int(local_size) + gap,
            left + int(local_size) + gap,
        ),
    ]


def apply_dba(
    images: torch.Tensor,
    *,
    size: int,
    part: int | None,
    value: float = 1.0,
) -> torch.Tensor:
    """Apply one DBA local trigger, or all four for the global test trigger."""

    _require_image_batch(images)
    output = images.clone()
    _, _, height, width = output.shape
    local_size = max(1, int(size) // 2)
    origins = _dba_origins(height, width, local_size)
    parts = range(4) if part is None else (int(part),)
    for trigger_part in parts:
        if trigger_part not in range(4):
            raise ValueError("DBA trigger part must be one of 0,1,2,3.")
        y0, x0 = origins[trigger_part]
        output[
            :,
            :,
            y0 : y0 + local_size,
            x0 : x0 + local_size,
        ] = float(value)
    return output


def blend_pattern_like(images: torch.Tensor) -> torch.Tensor:
    """Deterministic checker pattern in the same normalized range as inputs."""

    _require_image_batch(images)
    _, channels, height, width = images.shape
    yy = torch.arange(height, device=images.device).view(height, 1)
    xx = torch.arange(width, device=images.device).view(1, width)
    cell = max(2, min(height, width) // 8)
    checker = ((xx // cell + yy // cell) % 2).to(images.dtype)
    checker = checker * 2.0 - 1.0
    return checker.view(1, 1, height, width).expand(
        1, channels, height, width
    )


def apply_blend(images: torch.Tensor, *, alpha: float) -> torch.Tensor:
    _require_image_batch(images)
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("Blend alpha must be in (0, 1].")
    pattern = blend_pattern_like(images)
    return (
        (1.0 - float(alpha)) * images + float(alpha) * pattern
    ).clamp(-1.0, 1.0)


def dynamic_state(
    round_number: int,
    *,
    attack_start_round: int,
    period: int,
) -> tuple[str, float, float]:
    """Deterministic round-varying location, intensity, and size scale."""

    if int(period) < 1:
        raise ValueError("Dynamic period must be positive.")
    phase = (
        max(0, int(round_number) - int(attack_start_round)) // int(period)
    ) % 4
    locations = ("bottom-right", "top-left", "top-right", "center-edge")
    values = (1.0, 0.8, 0.6, 0.9)
    size_scales = (1.0, 0.75, 1.25, 1.0)
    return locations[phase], values[phase], size_scales[phase]


def apply_dynamic(
    images: torch.Tensor,
    *,
    size: int,
    round_number: int,
    attack_start_round: int,
    period: int,
) -> torch.Tensor:
    location, value, size_scale = dynamic_state(
        round_number,
        attack_start_round=attack_start_round,
        period=period,
    )
    dynamic_size = max(1, int(round(int(size) * float(size_scale))))
    return apply_badnets(
        images,
        size=dynamic_size,
        value=value,
        location=location,
    )
