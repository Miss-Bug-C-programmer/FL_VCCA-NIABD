from runtime_trace import (
    generate_runtime_trace,
    trace_fault_sequence,
)


def _profile():
    return {
        "name": "replay",
        "slow_client_fraction": 0.34,
        "normal_compute_slowdown_factor": 1.0,
        "slow_compute_slowdown_factor": 3.0,
        "normal_upload_delay_s": 0.01,
        "slow_upload_delay_s": 0.2,
        "availability_probability": 0.8,
        "upload_attempt_drop_probability": 0.2,
        "ack_delay_probability": 0.2,
        "ack_delay_s": 0.4,
        "events": {},
    }


def test_runtime_trace_replays_identical_external_conditions():
    first = generate_runtime_trace(
        profile=_profile(),
        seed=7,
        num_clients=3,
        rounds=4,
        warmup_rounds=1,
        participation_rate=0.75,
    )
    second = generate_runtime_trace(
        profile=_profile(),
        seed=7,
        num_clients=3,
        rounds=4,
        warmup_rounds=1,
        participation_rate=0.75,
    )

    assert list(trace_fault_sequence(first)) == list(
        trace_fault_sequence(second)
    )


def test_trace_is_strategy_independent_and_warmup_has_no_faults():
    baseline_trace = generate_runtime_trace(
        profile=_profile(),
        seed=9,
        num_clients=3,
        rounds=3,
        warmup_rounds=1,
        participation_rate=1.0,
    )
    vcaa_trace = generate_runtime_trace(
        profile=_profile(),
        seed=9,
        num_clients=3,
        rounds=3,
        warmup_rounds=1,
        participation_rate=1.0,
    )

    assert list(trace_fault_sequence(baseline_trace)) == list(
        trace_fault_sequence(vcaa_trace)
    )
    warmup = [
        event for event in baseline_trace.events
        if event.source_round == 1
    ]
    assert all(event.selected and event.available for event in warmup)
    assert all(event.compute_slowdown_factor == 1.0 for event in warmup)
    assert all(event.upload_delay_s == 0.0 for event in warmup)
    assert all(not event.dropped_attempts for event in warmup)
    assert all(not event.ack_delay_s_by_attempt for event in warmup)
