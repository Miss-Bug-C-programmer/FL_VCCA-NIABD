from pathlib import Path

import pandas as pd
import pytest

from scripts.merge_experiment_results import merge


def _write_summary(root: Path, *, strategy: str, extra: dict | None = None):
    path = root / "cifar10" / "badnets" / strategy / "seed_0"
    path.mkdir(parents=True)
    row = {
        "dataset": "cifar10",
        "attack_type": "badnets",
        "strategy": strategy,
        "seed": 0,
        "rounds": 1,
        "accuracy": 0.5,
    }
    row.update(extra or {})
    pd.DataFrame([row]).to_csv(
        path / "fedagg_run_summary_cifar10.csv",
        index=False,
    )


def test_merge_unions_old_and_new_columns_without_fabricating_values(tmp_path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _write_summary(old_root, strategy="baseline")
    _write_summary(
        new_root,
        strategy="vcaa",
        extra={"vcaa_algorithm_version": "vcaa-v3"},
    )

    outdir = tmp_path / "merged"
    result = merge([str(old_root), str(new_root)], str(outdir))

    assert result["fedagg_run_summary_cifar10.csv"] == 2
    frame = pd.read_csv(outdir / "fedagg_run_summary_cifar10.csv")
    assert "accuracy" in frame.columns
    assert "vcaa_algorithm_version" in frame.columns
    assert pd.isna(
        frame.loc[frame["strategy"] == "baseline", "vcaa_algorithm_version"]
    ).all()
    assert (
        frame.loc[frame["strategy"] == "vcaa", "vcaa_algorithm_version"]
        == "vcaa-v3"
    ).all()


def test_merge_rejects_duplicate_keys_by_default_and_supports_precedence(tmp_path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _write_summary(old_root, strategy="vcaa", extra={"accuracy": 0.1})
    _write_summary(new_root, strategy="vcaa", extra={"accuracy": 0.9})

    with pytest.raises(ValueError, match="duplicate result keys"):
        merge([str(old_root), str(new_root)], str(tmp_path / "error"))

    outdir = tmp_path / "newer"
    merge([str(old_root), str(new_root)], str(outdir), precedence="newer")
    frame = pd.read_csv(outdir / "fedagg_run_summary_cifar10.csv")
    assert len(frame) == 1
    assert frame.loc[0, "accuracy"] == 0.9
