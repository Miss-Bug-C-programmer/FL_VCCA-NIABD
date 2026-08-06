from __future__ import annotations

import copy
from dataclasses import replace

import pandas as pd
import pytest
import torch

from admission import TeacherKnowledge, TeacherMetadata
from checkpointing import (
    build_checkpoint_payload,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint_atomic,
)
from model_factory import architecture_assignment_hash, build_models
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense
from numeric_integrity import NumericIntegrityError
from result_schema import RESULT_SCHEMA_VERSION, schema_hash, validate_frame
from robust_aggregation import aggregate_probabilities


def _knowledge(count: int = 5, samples: int = 4, classes: int = 3):
    base = torch.arange(samples * classes, dtype=torch.float32).reshape(samples, classes)
    return [
        TeacherKnowledge(
            metadata=TeacherMetadata(
                client_id=index,
                model_round=1,
                generated_at_s=1.0,
                source_round=1,
                base_server_round=0,
                received_at_s=2.0,
                consumed_at_s=2.0,
                proxy_version="proxy-v3",
            ),
            logits=base + index * 0.01,
        )
        for index in range(count)
    ]


def test_probability_aggregators_are_simplex_and_fail_closed():
    teachers = [item.logits for item in _knowledge()]
    for rule in (
        "mean-soft-probabilities",
        "median-probabilities",
        "trimmed-mean-probabilities",
        "confidence-consistency-filtered-mean",
    ):
        result = aggregate_probabilities(
            teachers,
            method=rule,
            temperature=2.0,
            trim_fraction=0.1,
        )
        assert torch.isfinite(result).all()
        assert torch.allclose(result.sum(dim=1), torch.ones(result.shape[0]))
    bad = list(teachers)
    bad[0] = bad[0].clone()
    bad[0][0, 0] = float("nan")
    with pytest.raises(NumericIntegrityError):
        aggregate_probabilities(bad)


def test_niabd_chunking_is_equivalent_and_memory_state_is_proxy_bound():
    knowledge = _knowledge()
    config = NIABDConfig(
        warmup_rounds=1,
        minimum_consensus_teachers=3,
        proxy_chunk_size=2,
    )
    chunked = NeuroInspiredAdaptiveBackdoorDefense(config)
    full = NeuroInspiredAdaptiveBackdoorDefense(
        replace(config, proxy_chunk_size=0)
    )
    student = knowledge[0].logits.clone()
    first_chunked = chunked.purify(
        teacher_knowledge=knowledge,
        student_logits=student,
        proxy_labels=torch.zeros(4, dtype=torch.long),
        current_round=1,
    )
    first_full = full.purify(
        teacher_knowledge=knowledge,
        student_logits=student,
        proxy_labels=torch.zeros(4, dtype=torch.long),
        current_round=1,
    )
    assert first_chunked.metrics["niabd_memory_updated"] is True
    assert torch.equal(chunked.prototype_mean, full.prototype_mean)
    assert first_chunked.metrics["niabd_prototype_update_reason"] == first_full.metrics["niabd_prototype_update_reason"]


def test_niabd_warmup_freezes_when_consensus_is_not_established():
    knowledge = _knowledge(count=4)
    knowledge[-1] = TeacherKnowledge(
        metadata=knowledge[-1].metadata,
        logits=knowledge[-1].logits + 100.0,
    )
    defense = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(
            warmup_rounds=1,
            minimum_consensus_teachers=4,
            consensus_recovery_fraction=0.75,
        )
    )
    result = defense.purify(
        teacher_knowledge=knowledge,
        student_logits=knowledge[0].logits,
        proxy_labels=torch.zeros(4, dtype=torch.long),
        current_round=1,
    )
    assert result.metrics["niabd_defense_available"] is False
    assert result.metrics["niabd_purification_applied"] is False
    assert result.metrics["niabd_memory_updated"] is False
    assert result.metrics["niabd_prototype_update_reason"] in {
        "freeze_no_safe_candidate",
        "freeze_ambiguous_consensus",
    }


def test_checkpoint_roundtrip_and_mismatch_are_fail_closed(tmp_path):
    clients, server = build_models(
        "cifar10",
        2,
        "cpu",
        server_architecture="resnet18",
        client_architectures=["small_cnn", "mobilenet_v2"],
    )
    original = copy.deepcopy(server.state_dict())
    payload = build_checkpoint_payload(
        current_round=1,
        expected_rounds=2,
        run_uid="run-v3",
        config_sha256="config-hash",
        runtime="sync",
        server_model=server,
        client_models=clients,
        data_identity={"dataset": "cifar10"},
        proxy_identity={"version": "proxy-v3"},
        architecture_assignment={
            "hash": architecture_assignment_hash(
                server_architecture="resnet18",
                client_architectures=["small_cnn", "mobilenet_v2"],
            )
        },
    )
    path = tmp_path / "round_1.pt"
    save_checkpoint_atomic(payload, path)
    loaded = load_checkpoint(
        path,
        expected_config_sha256="config-hash",
        expected_runtime="sync",
        expected_rounds=2,
    )
    with pytest.raises(ValueError):
        load_checkpoint(path, expected_config_sha256="wrong")
    with torch.no_grad():
        for parameter in server.parameters():
            parameter.add_(1.0)
    restore_checkpoint(loaded, server_model=server, client_models=clients)
    assert all(torch.equal(server.state_dict()[key], value) for key, value in original.items())


def test_v3_schema_has_lineage_and_transaction_fields():
    frame = pd.DataFrame([{
        "run_uid": "run",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "vcaa_algorithm_version": "none",
        "niabd_algorithm_version": "none",
        "aggregation_algorithm_version": "aggregation-v1-probability-space",
        "run_class": "smoke",
        "attack_condition": "clean",
        "transaction_id": "tx",
        "transaction_status": "committed",
        "student_snapshot_sha256": None,
        "received_teachers": 0,
        "admitted_teachers": 0,
        "memory_candidate_teachers": None,
        "normal_eligible_teachers": None,
        "drift_recovery_candidates": None,
        "memory_update_teachers": None,
        "teachers_purified": 0,
        "niabd_defense_available": None,
        "niabd_purification_applied": None,
        "niabd_memory_updated": None,
        "niabd_memory_update_reason": None,
        "niabd_observations": None,
        "vcaa_history_size": None,
        "numeric_failure_count": 0,
        "rollback_reason": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
    }])
    validate_frame(frame)
    assert len(schema_hash()) == 64
