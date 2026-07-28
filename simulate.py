"""Backward-compatible imports for the former experiment simulator module."""

from federated_runtime import evaluate_with_loss, run_fedagg_server_client


simulate_fedagg_server_client = run_fedagg_server_client

__all__ = [
    "evaluate_with_loss",
    "run_fedagg_server_client",
    "simulate_fedagg_server_client",
]
