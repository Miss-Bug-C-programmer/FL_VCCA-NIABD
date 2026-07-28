import copy
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from admission import (
    AdmissionDecision,
    TeacherAdmissionRecord,
)
from federated_runtime import run_fedagg_server_client
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense
from vcaa import VCAAConfig, VersionContentAwareAdmission


class _TinyVisionModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


def _dataloaders(num_clients=2):
    generator = torch.Generator().manual_seed(7)
    x = torch.rand(8, 3, 8, 8, generator=generator) * 2.0 - 1.0
    y = torch.randint(0, 3, (8,), generator=generator)
    dataset = TensorDataset(x, y)
    return {
        "client": [
            DataLoader(dataset, batch_size=4, shuffle=False)
            for _ in range(num_clients)
        ],
        "proxy": DataLoader(dataset, batch_size=4, shuffle=False),
        "test": DataLoader(dataset, batch_size=4, shuffle=False),
    }


def test_one_round_runs_with_real_serialized_client_logits():
    clients = [_TinyVisionModel() for _ in range(2)]
    server = _TinyVisionModel()
    server_before = copy.deepcopy(server.state_dict())

    metrics = run_fedagg_server_client(
        clients,
        server,
        _dataloaders(),
        device="cpu",
        local_epochs=1,
        rounds=1,
        learning_rate=0.01,
        strict_numeric_checks=True,
    )

    assert metrics["topology"] == "server-client"
    assert metrics["num_clients"] == 2
    assert metrics["clients_trained"] == [2]
    assert metrics["server_client_distillations"] == [2]
    assert len(metrics["acc_list"]) == 1
    assert 0.0 <= metrics["acc_list"][0] <= 1.0
    assert metrics["nonfinite_distill_rollbacks"] == [0]
    assert metrics["knowledge_interface"] == "serialized-proxy-logits"
    assert metrics["client_upload_bytes"][0] > 0
    assert metrics["server_update_applied"] == [1]
    assert metrics["client_reverse_distillations"] == [2]
    assert metrics["vcaa_enabled"] == 0
    assert metrics["teachers_admitted"] == [2]
    assert metrics["teachers_rejected"] == [0]
    assert math.isnan(metrics["niabd_anomaly_fraction"][0])
    assert math.isnan(metrics["niabd_threshold_mean"][0])
    assert any(
        not torch.equal(server_before[key], server.state_dict()[key])
        for key in server_before
    )


def test_model_loader_cardinality_mismatch_is_rejected():
    try:
        run_fedagg_server_client(
            [_TinyVisionModel()],
            _TinyVisionModel(),
            _dataloaders(num_clients=2),
            rounds=1,
        )
    except ValueError as exc:
        assert "number of client models" in str(exc)
    else:
        raise AssertionError("Expected a client/model cardinality error.")


class _RejectSecondTeacher:
    name = "vcaa"

    def reset(self):
        return None

    def evaluate(self, **kwargs):
        del kwargs
        return AdmissionDecision(
            method="vcaa",
            threshold=0.5,
            admitted_client_ids=(0,),
            rejected_client_ids=(1,),
            records=(
                TeacherAdmissionRecord(
                    client_id=0,
                    admitted=True,
                    score=0.8,
                    components={
                        "version_score": 1.0,
                        "content_score": 0.6,
                    },
                ),
                TeacherAdmissionRecord(
                    client_id=1,
                    admitted=False,
                    score=0.2,
                    components={
                        "version_score": 0.0,
                        "content_score": 0.4,
                    },
                ),
            ),
        )


class _RejectAllTeachers:
    name = "vcaa"

    def reset(self):
        return None

    def evaluate(self, **kwargs):
        knowledge = kwargs["teacher_knowledge"]
        client_ids = tuple(
            int(item.metadata.client_id) for item in knowledge
        )
        return AdmissionDecision(
            method="vcaa",
            threshold=1.0,
            admitted_client_ids=(),
            rejected_client_ids=client_ids,
            records=tuple(
                TeacherAdmissionRecord(
                    client_id=client_id,
                    admitted=False,
                    score=0.0,
                )
                for client_id in client_ids
            ),
        )


def test_admission_controller_filters_client_uploads_before_aggregation():
    metrics = run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _dataloaders(),
        device="cpu",
        rounds=1,
        admission_controller=_RejectSecondTeacher(),
    )

    assert metrics["vcaa_enabled"] == 1
    assert metrics["admission_method"] == "vcaa"
    assert metrics["teachers_admitted"] == [1]
    assert metrics["teachers_rejected"] == [1]
    assert metrics["server_updates_from_clients"] == [1]
    assert metrics["teacher_utilization"] == [0.5]
    assert metrics["teacher_admission_records"][0][1]["admitted"] is False


def test_real_vcaa_controller_runs_inside_one_round_fedagg():
    controller = VersionContentAwareAdmission(
        VCAAConfig(warmup_rounds=1)
    )
    metrics = run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _dataloaders(),
        device="cpu",
        rounds=1,
        admission_controller=controller,
    )

    assert metrics["admission_method"] == "vcaa"
    assert metrics["teachers_admitted"] == [2]
    assert len(metrics["teacher_admission_records"][0]) == 2
    assert all(
        0.0 <= record["score"] <= 1.0
        for record in metrics["teacher_admission_records"][0]
    )


def test_niabd_only_runs_without_vcaa():
    defense = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1)
    )
    metrics = run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _dataloaders(),
        device="cpu",
        rounds=1,
        defense_controller=defense,
    )

    assert metrics["vcaa_enabled"] == 0
    assert metrics["niabd_enabled"] == 1
    assert metrics["defense_method"] == "niabd"
    assert metrics["teachers_purified"] == [2]
    assert metrics["niabd_warmup"] == [1.0]


def test_vcaa_and_niabd_can_be_enabled_together():
    admission = VersionContentAwareAdmission(
        VCAAConfig(warmup_rounds=1)
    )
    defense = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1)
    )
    metrics = run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _dataloaders(),
        device="cpu",
        rounds=1,
        admission_controller=admission,
        defense_controller=defense,
    )

    assert metrics["vcaa_enabled"] == 1
    assert metrics["niabd_enabled"] == 1
    assert metrics["teachers_admitted"] == [2]
    assert metrics["teachers_purified"] == [2]


def test_no_admitted_teacher_skips_niabd_and_server_update_cleanly():
    defense = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1)
    )
    metrics = run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _dataloaders(),
        device="cpu",
        rounds=1,
        admission_controller=_RejectAllTeachers(),
        defense_controller=defense,
    )

    assert metrics["teachers_admitted"] == [0]
    assert metrics["teachers_purified"] == [0]
    assert metrics["server_update_applied"] == [0]
    assert math.isnan(metrics["niabd_prototype_observations"][0])
