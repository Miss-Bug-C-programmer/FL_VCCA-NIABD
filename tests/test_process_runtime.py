import multiprocessing
import math
import os

import pytest
import torch

from data_utils import (
    build_client_dataloaders_from_plan,
    build_federated_data_plan,
    build_server_dataloaders_from_plan,
    cleanup_dataloaders,
)
from model_factory import build_model
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense
from process_runtime import (
    ProcessRuntimeConfig,
    _require_warmup_progress,
    _resolve_process_device,
    run_fedagg_server_client_process_async,
)
from round_coordinator import SemiAsyncRoundCoordinator
from runtime_trace import generate_runtime_trace
from vcaa import VCAAConfig, VersionContentAwareAdmission


def test_empty_process_warmup_fails_before_advancing_rounds():
    with pytest.raises(
        TimeoutError,
        match="warmup received no ClientLogitsPacket",
    ):
        _require_warmup_progress(
            warmup=True,
            dispatched_count=4,
            available_packets=0,
            hard_deadline_s=900.0,
        )


def test_process_cuda_device_without_index_resolves_to_visible_device_zero():
    assert _resolve_process_device("cuda") == torch.device("cuda:0")
    assert _resolve_process_device("cuda:2") == torch.device("cuda:2")
    assert _resolve_process_device("cpu") == torch.device("cpu")


def _write_femnist(root):
    generator = torch.Generator().manual_seed(31)
    train_x = torch.randint(
        0,
        256,
        (60, 28, 28),
        dtype=torch.uint8,
        generator=generator,
    )
    train_y = torch.randint(0, 62, (60,), generator=generator)
    test_x = torch.randint(
        0,
        256,
        (20, 28, 28),
        dtype=torch.uint8,
        generator=generator,
    )
    test_y = torch.randint(0, 62, (20,), generator=generator)
    torch.save({"x": train_x, "y": train_y}, root / "femnist_train.pt")
    torch.save({"x": test_x, "y": test_y}, root / "femnist_test.pt")


def _profile():
    return {
        "name": "integration-stale",
        "slow_client_fraction": 0.25,
        "normal_compute_slowdown_factor": 1.0,
        "slow_compute_slowdown_factor": 4.0,
        "normal_upload_delay_s": 0.0,
        "slow_upload_delay_s": 0.05,
        "availability_probability": 1.0,
        "upload_attempt_drop_probability": 0.0,
        "ack_delay_probability": 0.0,
        "ack_delay_s": 0.0,
        "events": {
            "0:1": {
                "ack_delay_s_by_attempt": {"1": 0.5},
            },
            "1:1": {
                "dropped_attempts": [1],
            },
        "2:2": {
            "compute_slowdown_factor": 1.5,
            "upload_delay_s": 5.0,
        },
        },
    }


def _consumed(event):
    return not (
        isinstance(event["version_lag"], float)
        and math.isnan(event["version_lag"])
    )


@pytest.fixture(scope="module")
def process_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("process-runtime-data")
    _write_femnist(root)
    plan = build_federated_data_plan(
        dataset_path=str(root),
        dataset_name="femnist",
        num_clients=3,
        batch_size=8,
        seed=0,
        partition_scheme="iid",
        label_skew_classes=2,
        quantity_skew_alpha=0.5,
        val_ratio=0.1,
        proxy_ratio=0.1,
        proxy_dataset_size=6,
    )
    server_loaders = build_server_dataloaders_from_plan(plan)
    trace = generate_runtime_trace(
        profile=_profile(),
        seed=0,
        num_clients=3,
        rounds=5,
        warmup_rounds=1,
        participation_rate=1.0,
    )
    config = ProcessRuntimeConfig(
        quorum_fraction=0.5,
        warmup_rounds=1,
        soft_deadline_factor=1.5,
        hard_deadline_factor=2.0,
        soft_deadline_override_s=2.0,
        hard_deadline_override_s=10.0,
        rpc_timeout_s=0.1,
        max_retries=4,
        retry_backoff_s=0.02,
        registration_timeout_s=60.0,
        shutdown_timeout_s=30.0,
        server_device="cpu",
        client_device="cpu",
        strict_numeric_checks=True,
    )
    torch.manual_seed(0)
    server_model = build_model(
        "resnet18",
        dataset_name="femnist",
        device="cpu",
    )
    try:
        metrics = run_fedagg_server_client_process_async(
            server_model=server_model,
            server_dataloaders=server_loaders,
            data_plan=plan,
            trace=trace,
            config=config,
            local_epochs=1,
            rounds=5,
            learning_rate=0.01,
            distill_temperature=2.0,
            admission_controller=VersionContentAwareAdmission(
                VCAAConfig(
                    warmup_rounds=1,
                    time_unit_s=1.0,
                )
            ),
            defense_controller=NeuroInspiredAdaptiveBackdoorDefense(
                NIABDConfig(warmup_rounds=1)
            ),
            enable_client_distillation=True,
        )
    finally:
        cleanup_dataloaders(server_loaders)
    return plan, trace, metrics


@pytest.mark.integration
def test_process_runtime_enforces_real_process_model_ownership(
    process_result,
):
    _, _, metrics = process_result
    parent_pid = int(metrics["parent_pid"])
    client_pids = {
        int(pid) for pid in metrics["client_pids"].values()
    }

    assert len(client_pids) == 3
    assert parent_pid == os.getpid()
    assert parent_pid not in client_pids
    assert metrics["server_has_client_model_refs"] is False
    assert metrics["all_clients_stopped"] is True
    assert not client_pids.intersection(
        child.pid for child in multiprocessing.active_children()
    )


@pytest.mark.integration
def test_real_client_inference_roundtrips_through_tcp(process_result):
    _, _, metrics = process_result
    events = metrics["runtime_events"]

    assert events
    assert all(event["logits_dtype"] == "float32" for event in events)
    assert all(
        event["payload_sha256"] == event["inference_sha256"]
        for event in events
    )
    assert all(event["payload_bytes"] > 0 for event in events)
    assert all(event["wire_bytes"] > event["payload_bytes"] for event in events)
    assert all(event["predict_logits_calls"] >= 1 for event in events)


def test_proxy_task_and_client_view_do_not_expose_admission_labels(
    process_result,
):
    plan, trace, metrics = process_result
    coordinator = SemiAsyncRoundCoordinator(
        trace=trace,
        proxy_version=plan.proxy_version,
        local_epochs=1,
        learning_rate=0.01,
        distillation_temperature=2.0,
        enable_client_distillation=False,
    )
    summary = coordinator.dispatch_round(
        server_round=1,
        latest_server_packet=None,
    )
    assert summary.dispatched_clients
    status, task = coordinator.get_task(client_id=0, pid=12345)
    assert status == "TASK"
    assert task is not None
    assert "label" not in str(task.rpc_metadata()).lower()
    assert "target" not in str(task.rpc_metadata()).lower()
    private_loader, proxy_input_loader = build_client_dataloaders_from_plan(
        plan,
        client_id=0,
    )
    try:
        proxy_batch = next(iter(proxy_input_loader))
        assert torch.is_tensor(proxy_batch)
        assert not isinstance(proxy_batch, (tuple, list))
    finally:
        cleanup_dataloaders({
            "private": private_loader,
            "proxy": proxy_input_loader,
        })
    assert all(
        not torch.isnan(torch.tensor(event["proxy_accuracy"]))
        for event in metrics["runtime_events"]
        if _consumed(event)
    )


@pytest.mark.integration
def test_real_slow_client_becomes_stale_and_reaches_vcaa(process_result):
    _, _, metrics = process_result
    stale = [
        event for event in metrics["runtime_events"]
        if _consumed(event) and int(event["version_lag"]) > 0
    ]
    injected_stale = [
        event for event in stale
        if float(event["injected_compute_delay_s"]) > 0.0
    ]
    fresh = [
        event for event in metrics["runtime_events"]
        if _consumed(event) and int(event["version_lag"]) == 0
    ]

    assert stale
    assert injected_stale
    late = injected_stale[0]
    assert int(late["consumed_round"]) > int(late["source_round"])
    assert late["received_at_s"] >= late["generated_at_s"]
    assert late["injected_compute_delay_s"] > 0.0
    assert late["transport_status"] in {"accepted", "duplicate"}
    assert not torch.isnan(torch.tensor(late["vcaa_version_score"]))
    assert not torch.isnan(torch.tensor(late["vcaa_content_score"]))
    assert not torch.isnan(torch.tensor(late["vcaa_final_score"]))
    assert any(
        event["vcaa_version_score"] != late["vcaa_version_score"]
        for event in fresh
    )
    assert any(
        not torch.isnan(torch.tensor(
            event["niabd_anomaly_fraction"]
        ))
        for event in fresh
        if event["admitted"] is True
    )


@pytest.mark.integration
def test_timeout_retry_and_attempt_drop_reuse_packet_identity(
    process_result,
):
    _, _, metrics = process_result
    timeout_event = next(
        event for event in metrics["runtime_events"]
        if int(event["client_id"]) == 0
        and int(event["source_round"]) == 1
    )
    drop_event = next(
        event for event in metrics["runtime_events"]
        if int(event["client_id"]) == 1
        and int(event["source_round"]) == 1
    )

    assert timeout_event["rpc_timeout_count"] == 1
    assert timeout_event["retry_count"] >= 1
    assert timeout_event["duplicate_receive_count"] == 1
    assert timeout_event["predict_logits_calls"] == 1
    assert timeout_event["local_train_count"] == 1
    assert (
        timeout_event["payload_sha256"]
        == timeout_event["inference_sha256"]
    )
    assert drop_event["upload_attempt_drop_count"] == 1
    assert drop_event["retry_count"] >= 1
    assert drop_event["upload_attempts"] >= 2
    assert drop_event["payload_sha256"] == drop_event["inference_sha256"]


@pytest.mark.integration
def test_busy_clients_do_not_receive_a_task_backlog(process_result):
    _, _, metrics = process_result
    assert sum(metrics["busy_skipped_clients"]) > 0
    task_ids = [event["task_id"] for event in metrics["runtime_events"]]
    packet_ids = [event["packet_id"] for event in metrics["runtime_events"]]
    assert len(task_ids) == len(set(task_ids))
    assert len(packet_ids) == len(set(packet_ids))


@pytest.mark.integration
def test_process_runtime_failure_cleans_up_other_children(tmp_path):
    _write_femnist(tmp_path)
    plan = build_federated_data_plan(
        dataset_path=str(tmp_path),
        dataset_name="femnist",
        num_clients=2,
        batch_size=8,
        seed=2,
        partition_scheme="iid",
        label_skew_classes=2,
        quantity_skew_alpha=0.5,
        val_ratio=0.1,
        proxy_ratio=0.1,
        proxy_dataset_size=4,
    )
    loaders = build_server_dataloaders_from_plan(plan)
    trace = generate_runtime_trace(
        profile=_profile(),
        seed=2,
        num_clients=2,
        rounds=1,
        warmup_rounds=1,
        participation_rate=1.0,
    )
    config = ProcessRuntimeConfig(
        registration_timeout_s=10.0,
        shutdown_timeout_s=5.0,
        server_device="cpu",
        client_device="cuda:999",
    )
    before = {child.pid for child in multiprocessing.active_children()}
    try:
        with pytest.raises(RuntimeError, match="Client process"):
            run_fedagg_server_client_process_async(
                server_model=build_model(
                    "resnet18",
                    dataset_name="femnist",
                    device="cpu",
                ),
                server_dataloaders=loaders,
                data_plan=plan,
                trace=trace,
                config=config,
                local_epochs=1,
                rounds=1,
                learning_rate=0.01,
                distill_temperature=2.0,
            )
    finally:
        cleanup_dataloaders(loaders)
    after = {child.pid for child in multiprocessing.active_children()}
    assert after == before
