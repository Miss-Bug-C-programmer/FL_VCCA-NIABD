import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import contextlib

from numeric_integrity import NumericIntegrityError
from simulate import evaluate_with_loss
import trainer as trainer_module
from trainer import distill_with_logits, local_train


class _BadLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        return torch.full(
            (inputs.shape[0], 2),
            float("nan"),
            device=inputs.device,
        ) * self.scale


def _loader():
    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(inputs, labels), batch_size=2)


def test_local_training_strict_mode_rejects_nonfinite_logits():
    with pytest.raises(NumericIntegrityError):
        local_train(
            _BadLogits(),
            _loader(),
            device="cpu",
            strict_numeric_checks=True,
        )


def test_evaluation_reports_and_sanitizes_nonfinite_batches():
    accuracy, loss, nonfinite = evaluate_with_loss(
        _BadLogits(),
        _loader(),
        device="cpu",
    )
    assert 0.0 <= accuracy <= 1.0
    assert torch.isfinite(torch.tensor(loss))
    assert nonfinite == 2


class _ControlledGradient(torch.autograd.Function):
    calls = 0
    always_nonfinite = False

    @staticmethod
    def forward(ctx, inputs, weight):
        ctx.save_for_backward(inputs, weight)
        return inputs * weight

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        _ControlledGradient.calls += 1
        grad_inputs = grad_output * weight
        grad_weight = (grad_output * inputs).sum()
        if (
            _ControlledGradient.always_nonfinite
            or _ControlledGradient.calls == 1
        ):
            grad_weight = grad_weight * float("nan")
        return grad_inputs, grad_weight


class _ControlledGradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return _ControlledGradient.apply(inputs, self.weight)


class _TestGradScaler:
    def __init__(self):
        self._scale = 16.0
        self._found_inf = False

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        self._found_inf = any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all().item())
            for group in optimizer.param_groups
            for parameter in group["params"]
        )

    def step(self, optimizer):
        if not self._found_inf:
            optimizer.step()

    def update(self):
        if self._found_inf:
            self._scale /= 2.0
        self._found_inf = False


def _controlled_loader():
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
    )
    labels = torch.tensor([0, 1, 1, 0])
    return DataLoader(
        TensorDataset(inputs, labels),
        batch_size=2,
        shuffle=False,
    )


def _enable_test_amp(monkeypatch):
    monkeypatch.setattr(
        trainer_module,
        "use_amp_for_device",
        lambda _device: True,
    )
    monkeypatch.setattr(
        trainer_module,
        "_amp_context",
        lambda _device, enabled=True: contextlib.nullcontext(),
    )


def test_amp_overflow_skips_step_backs_off_and_recovers(monkeypatch):
    _enable_test_amp(monkeypatch)
    _ControlledGradient.calls = 0
    _ControlledGradient.always_nonfinite = False
    model = _ControlledGradientModel()
    scaler = _TestGradScaler()
    stats = {}

    local_train(
        model,
        _controlled_loader(),
        device="cpu",
        amp=True,
        strict_numeric_checks=True,
        scaler=scaler,
        numeric_stats=stats,
    )

    assert stats["amp_overflow_count"] == 1
    assert stats["optimizer_step_skipped_count"] == 1
    assert stats["optimizer_step_count"] == 1
    assert scaler.get_scale() == 8.0
    assert all(torch.isfinite(value).all() for value in model.parameters())


def test_amp_overflow_streak_remains_fail_fast(monkeypatch):
    _enable_test_amp(monkeypatch)
    _ControlledGradient.calls = 0
    _ControlledGradient.always_nonfinite = True

    with pytest.raises(
        NumericIntegrityError,
        match="did not recover",
    ):
        local_train(
            _ControlledGradientModel(),
            _controlled_loader(),
            device="cpu",
            amp=True,
            strict_numeric_checks=True,
            scaler=_TestGradScaler(),
            max_consecutive_amp_overflows=1,
        )


def test_non_amp_nonfinite_gradient_remains_fail_fast():
    _ControlledGradient.calls = 0
    _ControlledGradient.always_nonfinite = True

    with pytest.raises(
        NumericIntegrityError,
        match="Non-finite local training gradient",
    ):
        local_train(
            _ControlledGradientModel(),
            _controlled_loader(),
            device="cpu",
            amp=False,
            strict_numeric_checks=True,
        )


def test_amp_distillation_overflow_skips_step_and_recovers(monkeypatch):
    _enable_test_amp(monkeypatch)
    _ControlledGradient.calls = 0
    _ControlledGradient.always_nonfinite = False
    model = _ControlledGradientModel()
    scaler = _TestGradScaler()
    stats = {}

    distill_with_logits(
        model,
        _controlled_loader(),
        torch.tensor(
            [[2.0, -1.0], [-1.0, 2.0], [-1.0, 2.0], [2.0, -1.0]]
        ),
        device="cpu",
        amp=True,
        strict_numeric_checks=True,
        scaler=scaler,
        numeric_stats=stats,
    )

    assert stats["amp_overflow_count"] == 1
    assert stats["optimizer_step_skipped_count"] == 1
    assert stats["optimizer_step_count"] == 1
    assert scaler.get_scale() == 8.0
    assert all(torch.isfinite(value).all() for value in model.parameters())
