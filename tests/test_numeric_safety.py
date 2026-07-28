import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from numeric_integrity import NumericIntegrityError
from simulate import evaluate_with_loss
from trainer import local_train


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
