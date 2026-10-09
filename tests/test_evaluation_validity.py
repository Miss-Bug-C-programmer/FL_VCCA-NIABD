"""Invalid effect metrics must never become ordinary statistical observations."""

import math
import sys

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attacks import AttackConfig, AttackPlan
from attacks.evaluation import evaluate_aa, evaluate_basr, evaluate_ta
from experiment_runner import _round_rows, _summary_row
from tests.test_runner_outputs import _metrics
from tests.test_summarize_results import rows_for_missing_process_summary


class MixedLogits(nn.Module):
    def __init__(self, bad_value):
        super().__init__()
        self.calls = 0
        self.bad_value = bad_value

    def forward(self, images):
        self.calls += 1
        if self.calls == 2:
            return torch.full((len(images), 2), self.bad_value)
        return torch.tensor([1.0, 0.0]).repeat(len(images), 1)


def _plan(attack="badnets"):
    return AttackPlan.build(
        seed=0, num_clients=1,
        config=AttackConfig(
            attack_type=attack, attack_start_round=1,
            attack_end_round=1, trigger_size=2,
        ),
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("metric", ["ta", "aa"])
def test_mixed_nonfinite_evaluation_has_no_reportable_value(bad_value, strict, metric):
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.full((4,), 0 if metric == "ta" else 1)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    kwargs = dict(device="cpu", strict_numeric_checks=strict)
    if metric == "aa":
        kwargs.update(plan=_plan(), round_number=1)
    evaluate = evaluate_ta if metric == "ta" else evaluate_aa
    result = evaluate(MixedLogits(bad_value), loader, **kwargs)
    assert not result.valid
    assert math.isnan(result.value)
    assert result.nonfinite_batches == 1
    assert result.reason == "nonfinite_logits"
    # Partial/sanitized counts remain diagnostics, never a reportable ratio.
    assert result.denominator == (2 if strict else 4)


def test_legacy_basr_alias_also_invalidates_nonfinite_evaluation():
    loader = DataLoader(
        TensorDataset(torch.zeros(4, 3, 8, 8), torch.ones(4, dtype=torch.long)),
        batch_size=2,
    )
    result = evaluate_basr(
        MixedLogits(float("nan")), loader, device="cpu",
        plan=_plan(), round_number=1,
    )
    assert math.isnan(result.basr)


def test_finite_evaluation_preserves_real_zero_aa_and_full_denominators():
    class AlwaysNonTarget(nn.Module):
        def forward(self, images):
            return torch.tensor([0.0, 1.0]).repeat(len(images), 1)

    loader = DataLoader(
        TensorDataset(torch.zeros(4, 3, 8, 8), torch.ones(4, dtype=torch.long)),
        batch_size=2,
    )
    ta = evaluate_ta(AlwaysNonTarget(), loader, device="cpu", strict_numeric_checks=True)
    aa = evaluate_aa(AlwaysNonTarget(), loader, device="cpu", plan=_plan(),
                     round_number=1, strict_numeric_checks=True)
    assert ta.valid and ta.value == 1.0 and ta.denominator == 4
    assert aa.valid and aa.value == 0.0 and aa.denominator == 4


@pytest.mark.parametrize("metric", ["ta", "aa"])
def test_zero_evaluation_denominator_is_explicitly_invalid(metric):
    loader = DataLoader(
        TensorDataset(torch.zeros(0, 3, 8, 8), torch.empty(0, dtype=torch.long)),
        batch_size=2,
    )
    kwargs = dict(device="cpu")
    if metric == "aa":
        kwargs.update(plan=_plan(), round_number=1)
    result = (evaluate_ta if metric == "ta" else evaluate_aa)(MixedLogits(0.0), loader, **kwargs)
    assert not result.valid and math.isnan(result.value)
    assert result.reason == ("empty_test_set" if metric == "ta" else "no_eligible_samples")


@pytest.mark.parametrize("flag", [False, "False", "0", "0.0", "unexpected"])
def test_mask_blocks_invalid_finite_values_and_both_aliases(flag):
    from result_schema import mask_invalid_accuracy_frame

    frame = pd.DataFrame([{
        "final_ta": 0.9, "final_accuracy": 0.9, "final_ta_valid": flag,
        "final_aa": 0.8, "final_basr_global": 0.8, "final_aa_valid": flag,
        "final_ta_numerator": 9, "final_ta_denominator": 10,
    }])
    result = mask_invalid_accuracy_frame(frame)
    assert result[["final_ta", "final_accuracy", "final_aa", "final_basr_global"]].isna().all().all()
    assert result.loc[0, "final_ta_numerator"] == 9
    assert frame.loc[0, "final_ta"] == 0.9  # Never mutate the input/history.


def test_nonfinite_counter_overrides_true_validity_and_legacy_values_survive():
    from result_schema import mask_invalid_accuracy_frame

    frame = pd.DataFrame([
        {"final_ta": 0.7, "final_ta_valid": True, "final_ta_nonfinite_batches": 1},
        {"final_ta": 0.6, "final_ta_valid": "true", "final_ta_nonfinite_batches": 0},
        {"final_ta": 0.5},
    ])
    result = mask_invalid_accuracy_frame(frame)
    assert pd.isna(result.loc[0, "final_ta"])
    assert result.loc[1, "final_ta"] == 0.6
    assert result.loc[2, "final_ta"] == 0.5


def _invalid_round_rows():
    metrics = _metrics()
    metrics.update({
        "attack_type": "badnets", "attack_active": [1, 1],
        "ta": [0.4, 0.99], "ta_valid": [True, False],
        "ta_numerator": [4, 9], "ta_denominator": [10, 9],
        "ta_nonfinite_batches": [0, 1],
        "basr_global": [0.3, 0.98], "aa": [0.3, 0.98],
        "aa_valid": [True, False], "aa_nonfinite_batches": [0, 1],
        "aa_invalid_reason": ["", "nonfinite_logits"],
        "aa_numerator": [3, 9], "aa_denominator": [10, 9],
    })
    return list(_round_rows(
        metrics, run_uid="invalid-last", dataset_name="cifar10",
        seed=0, num_clients=3, partition_scheme="iid",
    ))


def test_round_export_and_summary_never_replace_invalid_final_with_earlier_round():
    rows = _invalid_round_rows()
    assert pd.isna(rows[-1]["ta"]) and pd.isna(rows[-1]["accuracy"])
    assert pd.isna(rows[-1]["aa"]) and pd.isna(rows[-1]["basr_global"])
    assert rows[-1]["ta_numerator"] == 9
    summary = _summary_row(rows)
    for key in ("final_ta", "final_accuracy", "final_aa", "final_basr_global"):
        assert pd.isna(summary[key])
    assert summary["final_ta_valid"] is False
    assert summary["final_aa_valid"] is False
    assert summary["final_ta_nonfinite_batches"] == 1
    assert summary["final_aa_invalid_reason"] == "nonfinite_logits"
    assert summary["best_ta"] == summary["best_accuracy"] == 0.4
    assert summary["mean_attack_window_basr"] == 0.3
    assert pd.isna(summary["attack_window_basr_auc"])
    assert summary["ta_valid_rounds"] == summary["aa_valid_rounds"] == 1


def test_summary_defensively_masks_direct_unsanitized_rows():
    rows = _invalid_round_rows()
    rows[-1].update(ta=0.99, accuracy=0.99, aa=0.98, basr_global=0.98)
    summary = _summary_row(rows)
    assert pd.isna(summary["final_ta"])
    assert pd.isna(summary["final_aa"])
    assert summary["best_accuracy"] == 0.4


def test_summary_all_invalid_has_no_best_value():
    rows = _invalid_round_rows()
    for row in rows:
        row.update(ta_valid=False, aa_valid=False)
    summary = _summary_row(rows)
    assert pd.isna(summary["best_ta"])
    assert pd.isna(summary["mean_attack_window_basr"])


def test_cross_seed_summary_masks_old_finite_invalid_rows_and_counts_observations(tmp_path):
    from summarize_results import summarize

    valid = rows_for_missing_process_summary()
    valid.update(final_ta=0.5, final_aa=0.2, final_basr_global=0.2,
                 final_ta_valid=True, final_aa_valid=True)
    invalid = {**valid, "seed": 1, "final_ta": 0.99, "final_accuracy": 0.99,
               "final_aa": 0.99, "final_basr_global": 0.99,
               "final_ta_valid": "False", "final_aa_valid": "False"}
    pd.DataFrame([valid, invalid]).to_csv(tmp_path / "fedagg_run_summary_cifar10.csv", index=False)
    result = summarize(str(tmp_path)).iloc[0]
    assert result["runs"] == 2
    assert result["final_ta_mean"] == result["final_accuracy_mean"] == 0.5
    assert result["final_aa_mean"] == 0.2
    assert result["final_ta_n"] == result["final_aa_n"] == 1


def test_statistics_cli_excludes_invalid_values_from_ci_and_pairs(tmp_path, monkeypatch):
    from scripts.compute_statistics import main

    rows = []
    for seed in range(3):
        rows.extend([
            dict(dataset="cifar10", seed=seed, strategy="baseline", final_ta=0.4, final_ta_valid=True),
            dict(dataset="cifar10", seed=seed, strategy="niabd", final_ta=0.5 if seed < 2 else 0.99, final_ta_valid=seed < 2),
        ])
    source, target = tmp_path / "input.csv", tmp_path / "statistics.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    monkeypatch.setattr(sys, "argv", ["compute_statistics", "--summary", str(source),
                                    "--metric", "final_ta", "--paired-method", "baseline", "--out", str(target)])
    main()
    result = pd.read_csv(target)
    assert result.loc[0, "n"] == 5
    assert result.loc[1, "n"] == 2
    assert result.loc[1, "mean"] == pytest.approx(0.1)


def test_merge_masks_values_without_rewriting_source_csv(tmp_path):
    from scripts.merge_experiment_results import merge

    source = tmp_path / "source"
    source.mkdir()
    path = source / "fedagg_run_summary_cifar10.csv"
    pd.DataFrame([dict(dataset="cifar10", attack_type="badnets", strategy="niabd", seed=0,
                       final_aa=0.9, final_basr_global=0.9, final_aa_valid=False)]).to_csv(path, index=False)
    before = path.read_bytes()
    target = tmp_path / "merged"
    merge([str(source)], str(target))
    result = pd.read_csv(target / path.name)
    assert pd.isna(result.loc[0, "final_aa"])
    assert pd.isna(result.loc[0, "final_basr_global"])
    assert path.read_bytes() == before


def test_formal_collector_excludes_invalid_final_values(tmp_path, monkeypatch):
    from scripts.collect_main_backdoor_results import main

    row = dict(dataset="cifar10", attack_type="badnets", strategy="niabd", seed=0,
               runtime="sync", niabd_algorithm_version="niabd-v3",
               result_schema_version="fedagg-results-v3", final_accuracy=0.5,
               final_basr_global=0.2, final_ta_valid=True, final_aa_valid=True,
               mean_attack_window_basr=0.3, peak_attack_window_basr=0.4,
               attack_window_basr_auc=0.6, post_attack_recovery_basr=0.2,
               total_poisoned_samples=3)
    invalid = {**row, "seed": 1, "final_accuracy": 0.99,
               "final_basr_global": 0.99, "final_ta_valid": False, "final_aa_valid": False}
    pd.DataFrame([row, invalid]).to_csv(tmp_path / "fedagg_run_summary_cifar10.csv", index=False)
    target = tmp_path / "collected.csv"
    monkeypatch.setattr(sys, "argv", ["collect", "--indir", str(tmp_path), "--out", str(target)])
    main()
    result = pd.read_csv(target).iloc[0]
    assert result["runs"] == 2
    assert result["ta_observations"] == result["aa_observations"] == 1
    assert result["clean_acc_mean"] == 0.5
    assert result["basr_mean"] == 0.2


def test_mixed_legacy_and_new_summary_files_keep_legacy_aliases(tmp_path):
    from summarize_results import summarize

    legacy = rows_for_missing_process_summary()
    legacy.update(final_basr_global=0.3)
    pd.DataFrame([legacy]).to_csv(tmp_path / "fedagg_run_summary_old.csv", index=False)
    current = {**legacy, "seed": 1, "final_ta": 0.7, "final_accuracy": 0.7,
               "final_aa": 0.1, "final_basr_global": 0.1,
               "final_ta_valid": True, "final_aa_valid": True}
    pd.DataFrame([current]).to_csv(tmp_path / "fedagg_run_summary_new.csv", index=False)
    result = summarize(str(tmp_path)).iloc[0]
    assert result["final_ta_n"] == result["final_aa_n"] == 2
    assert result["final_ta_mean"] == pytest.approx(0.6)
    assert result["final_aa_mean"] == pytest.approx(0.2)


def test_validator_rejects_invalid_finite_effect_metric():
    from result_schema import validate_accuracy_metrics

    with pytest.raises(ValueError, match="invalid.*TA|TA.*invalid"):
        validate_accuracy_metrics(pd.DataFrame([dict(ta=0.9, accuracy=0.9, ta_valid=False)]))
    with pytest.raises(ValueError, match="aliases"):
        validate_accuracy_metrics(pd.DataFrame([dict(aa=float("nan"), basr_global=0.9)]))
    validate_accuracy_metrics(pd.DataFrame([dict(ta=float("nan"), accuracy=float("nan"), ta_valid=False)]))
