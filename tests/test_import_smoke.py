import importlib


def test_core_modules_import_without_legacy_three_tier_modules():
    for module_name in (
        "admission",
        "data_utils",
        "defense",
        "device_utils",
        "distillation",
        "experiment_runner",
        "federated_client",
        "federated_runtime",
        "federated_server",
        "logits_transport",
        "model_factory",
        "models",
        "numeric_integrity",
        "niabd",
        "process_runtime",
        "round_coordinator",
        "rpc_transport",
        "runtime_trace",
        "simulate",
        "summarize_results",
        "trainer",
        "vcaa",
    ):
        assert importlib.import_module(module_name) is not None
