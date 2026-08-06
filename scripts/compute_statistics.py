"""Student-t confidence intervals and paired run statistics for v3 summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from scipy import stats
except ImportError:  # pragma: no cover - lightweight research environment
    stats = None


_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _ci(values: np.ndarray, confidence: float) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    if stats is not None:
        critical = float(
            stats.t.ppf((1.0 + confidence) / 2.0, values.size - 1)
        )
    elif abs(float(confidence) - 0.95) < 1e-9:
        critical = _T95.get(
            min(values.size - 1, 30),
            1.96,
        )
    else:
        critical = 1.96
    half = critical
    half *= float(values.std(ddof=1)) / np.sqrt(values.size)
    return mean, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metric", default="final_accuracy")
    parser.add_argument("--paired-method", default="")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out", default="statistics.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    if args.metric not in frame.columns:
        raise SystemExit(f"missing metric: {args.metric}")
    values = pd.to_numeric(frame[args.metric], errors="coerce").to_numpy()
    mean, lower, upper = _ci(values, float(args.confidence))
    output = [{
        "metric": args.metric,
        "n": int(np.isfinite(values).sum()),
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
    }]
    if args.paired_method:
        if "strategy" not in frame.columns or "seed" not in frame.columns:
            raise SystemExit("paired statistics require strategy and seed columns")
        baseline = frame[frame["strategy"] == args.paired_method]
        other = frame[frame["strategy"] != args.paired_method]
        joined = baseline.merge(other, on=["dataset", "seed"], suffixes=("_base", "_other"))
        left = pd.to_numeric(joined[f"{args.metric}_base"], errors="coerce")
        right = pd.to_numeric(joined[f"{args.metric}_other"], errors="coerce")
        difference = (right - left).dropna().to_numpy()
        if difference.size >= 2:
            if stats is not None:
                p_value = float(stats.ttest_1samp(difference, 0.0).pvalue)
            else:
                p_value = float("nan")
            output.append({
                "metric": f"paired_delta_vs_{args.paired_method}",
                "n": int(difference.size),
                "mean": float(difference.mean()),
                "ci_lower": float(_ci(difference, float(args.confidence))[1]),
                "ci_upper": float(_ci(difference, float(args.confidence))[2]),
                "p_value": p_value,
            })
    pd.DataFrame(output).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
