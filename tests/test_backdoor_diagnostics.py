from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from attacks import AttackConfig, AttackPlan
from federated_client import FederatedClient
from federated_runtime import run_fedagg_server_client
from logits_transport import ClientLogitsPacket
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense
from vcaa import VCAAConfig, VersionContentAwareAdmission

from test_backdoor_runtime import _loaders, _TinyVisionModel


def _diagnostic_plan(attack_type: str = "dynamic") -> AttackPlan:
    return AttackPlan.build(
        seed=11,
        num_clients=2,
        config=AttackConfig(
            attack_type=attack_type,
            target_label=0,
            malicious_fraction=0.5,
            poison_ratio=0.0,
            attack_start_round=1,
            attack_end_round=2,
            trigger_size=2,
            dynamic_period=1,
        ),
    )


def test_diagnostics_disabled_does_not_call_client_or_change_algorithm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled diagnostics must not execute")

    monkeypatch.setattr(
        FederatedClient,
        "compute_backdoor_diagnostics",
        forbidden,
    )
    metrics = run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _loaders(),
        device="cpu",
        rounds=1,
        local_epochs=1,
        attack_plan=_diagnostic_plan(),
        enable_backdoor_diagnostics=False,
    )

    assert metrics["backdoor_diagnostics_enabled"] == 0
    assert all(
        "clean_proxy_target_probability" not in row
        for row in metrics["backdoor_client_records"][0]
    )


def test_enabled_diagnostics_preserve_formal_metrics_and_model_state():
    torch.manual_seed(2026)
    base_clients = [_TinyVisionModel(), _TinyVisionModel()]
    base_server = _TinyVisionModel()
    disabled_clients = deepcopy(base_clients)
    enabled_clients = deepcopy(base_clients)
    disabled_server = deepcopy(base_server)
    enabled_server = deepcopy(base_server)
    plan = _diagnostic_plan()

    torch.manual_seed(77)
    disabled = run_fedagg_server_client(
        disabled_clients,
        disabled_server,
        _loaders(),
        device="cpu",
        rounds=2,
        local_epochs=1,
        attack_plan=plan,
        enable_backdoor_diagnostics=False,
    )
    torch.manual_seed(77)
    enabled = run_fedagg_server_client(
        enabled_clients,
        enabled_server,
        _loaders(),
        device="cpu",
        rounds=2,
        local_epochs=1,
        attack_plan=plan,
        enable_backdoor_diagnostics=True,
        backdoor_diagnostics_dataset="cifar10",
    )

    for key in (
        "acc_list",
        "loss_list",
        "basr_global",
        "poisoned_samples",
        "server_update_applied",
        "teachers_admitted",
        "teachers_rejected",
    ):
        assert enabled[key] == pytest.approx(disabled[key])
    for enabled_value, disabled_value in zip(
        enabled_server.state_dict().values(),
        disabled_server.state_dict().values(),
    ):
        assert torch.equal(enabled_value, disabled_value)

    records = enabled["backdoor_client_records"]
    assert len(records) == 2
    assert all(
        row["diagnostic_scope"] == "experiment-only oracle diagnostic"
        and row["diagnostic_usage"] == "not a deployable defense signal"
        and row["diagnostic_reporter_trust"] == "not assumed"
        for round_records in records
        for row in round_records
    )
    assert records[0][0]["diagnostic_seed"] != records[1][0]["diagnostic_seed"]
    assert records[0][0]["diagnostic_seed"] != records[0][1]["diagnostic_seed"]


def test_diagnostics_run_only_after_vcaa_and_niabd(monkeypatch):
    events: list[str] = []
    original_evaluate = VersionContentAwareAdmission.evaluate
    original_purify = NeuroInspiredAdaptiveBackdoorDefense.purify
    original_diagnostic = FederatedClient.compute_backdoor_diagnostics

    def tracked_evaluate(self, **kwargs):
        events.append("vcaa")
        assert all(
            set(vars(item)) == {"metadata", "logits"}
            for item in kwargs["teacher_knowledge"]
        )
        return original_evaluate(self, **kwargs)

    def tracked_purify(self, **kwargs):
        events.append("niabd")
        assert all(
            set(vars(item)) == {"metadata", "logits"}
            for item in kwargs["teacher_knowledge"]
        )
        return original_purify(self, **kwargs)

    def tracked_diagnostic(self, *args, **kwargs):
        events.append("diagnostic")
        return original_diagnostic(self, *args, **kwargs)

    monkeypatch.setattr(
        VersionContentAwareAdmission,
        "evaluate",
        tracked_evaluate,
    )
    monkeypatch.setattr(
        NeuroInspiredAdaptiveBackdoorDefense,
        "purify",
        tracked_purify,
    )
    monkeypatch.setattr(
        FederatedClient,
        "compute_backdoor_diagnostics",
        tracked_diagnostic,
    )
    run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _loaders(),
        device="cpu",
        rounds=1,
        local_epochs=1,
        admission_controller=VersionContentAwareAdmission(
            VCAAConfig(warmup_rounds=1)
        ),
        defense_controller=NeuroInspiredAdaptiveBackdoorDefense(
            NIABDConfig(warmup_rounds=1)
        ),
        attack_plan=_diagnostic_plan("badnets"),
        enable_backdoor_diagnostics=True,
        backdoor_diagnostics_dataset="cifar10",
    )

    assert events[:2] == ["vcaa", "niabd"]
    assert events[2:] == ["diagnostic", "diagnostic"]


def test_client_logits_packet_schema_has_no_diagnostic_or_attack_oracle_fields():
    packet_fields = tuple(field.name for field in fields(ClientLogitsPacket))
    forbidden = {
        "is_malicious",
        "attack_type",
        "trigger_id",
        "target_label",
        "poisoned_samples",
        "basr",
        "clean_proxy_target_probability",
        "triggered_proxy_target_probability",
        "clean_trigger_logit_l1_deviation",
        "clean_trigger_logit_l2_deviation",
        "clean_trigger_prediction_flip_rate",
    }

    assert forbidden.isdisjoint(packet_fields)
