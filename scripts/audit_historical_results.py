from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _last_finite(frame: pd.DataFrame, column: str) -> float:
    values = _number(frame, column).dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def _mean_finite(frame: pd.DataFrame, column: str) -> float:
    values = _number(frame, column).dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _longest_false(values: Iterable[bool]) -> int:
    longest = current = 0
    for value in values:
        if not bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _attack_window(frame: pd.DataFrame) -> pd.DataFrame:
    if "attack_active" in frame.columns:
        active = _number(frame, "attack_active")
        rows = frame.loc[active == 1].copy()
        if not rows.empty:
            return rows
    attack_type = str(frame.get("attack_type", pd.Series(["none"])).iloc[0])
    if attack_type == "none":
        return frame.iloc[0:0].copy()
    start = int(_last_finite(frame, "attack_start_round"))
    end = int(_last_finite(frame, "rounds"))
    rounds = _number(frame, "round")
    return frame.loc[(rounds >= start) & (rounds <= end)].copy()


def _auc(frame: pd.DataFrame, column: str) -> tuple[float, float, str]:
    if column not in frame.columns or "round" not in frame.columns:
        return float("nan"), float("nan"), "不可重建"
    values = frame[["round", column]].copy()
    values["round"] = pd.to_numeric(values["round"], errors="coerce")
    values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna().sort_values("round")
    if len(values) < 2:
        return float("nan"), float("nan"), "不可重建"
    rounds = values["round"].to_numpy(dtype=float)
    if not np.all(np.diff(rounds) == 1):
        return float("nan"), float("nan"), "不可重建（round 不连续）"
    raw = float(np.trapz(values[column].to_numpy(dtype=float), rounds))
    width = float(rounds[-1] - rounds[0])
    normalized = raw / width if width > 0 else float("nan")
    return raw, normalized, "可重建"


def _run_metrics(summary_path: Path) -> Dict[str, Any]:
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return {
            "summary_path": str(summary_path),
            "raw_round_csv": "不可重建",
            "dataset": "unknown",
            "error": "summary CSV 为空",
        }
    dataset = str(summary.get("dataset", pd.Series(["unknown"])).iloc[0])
    raw_path = summary_path.with_name(
        f"fedagg_experiment_results_{dataset}.csv"
    )
    if not raw_path.is_file():
        return {
            "summary_path": str(summary_path),
            "raw_round_csv": "不可重建",
            "dataset": dataset,
            "error": "对应 round CSV 不存在",
        }
    rounds = pd.read_csv(raw_path)
    rounds = rounds.sort_values("round") if "round" in rounds else rounds
    attack = _attack_window(rounds)
    raw_auc, normalized_auc, auc_status = _auc(attack, "basr_global")
    peak = _number(attack, "basr_global").dropna()
    peak_basr = float(peak.max()) if not peak.empty else float("nan")
    peak_round = (
        int(attack.loc[_number(attack, "basr_global").idxmax(), "round"])
        if not peak.empty and "round" in attack.columns
        else None
    )
    prototype = _number(rounds, "niabd_prototype_observations")
    updates = _number(rounds, "niabd_prototype_updated")
    eligible = _number(rounds, "niabd_memory_eligible_teachers")
    update_flags = updates.fillna(0).astype(float) > 0
    admission_denominator = _number(rounds, "teachers_admitted") + _number(
        rounds, "teachers_rejected"
    )
    admission_rate = _number(rounds, "teachers_admitted") / admission_denominator
    result: Dict[str, Any] = {
        "summary_path": str(summary_path),
        "raw_round_csv": str(raw_path),
        "run_uid": str(summary.get("run_uid", pd.Series(["unknown"])).iloc[0]),
        "dataset": dataset,
        "seed": int(_last_finite(summary, "seed")) if not math.isnan(_last_finite(summary, "seed")) else None,
        "runtime": str(summary.get("runtime", pd.Series(["unknown"])).iloc[0]),
        "strategy": str(summary.get("strategy", pd.Series(["unknown"])).iloc[0]),
        "attack_type": str(summary.get("attack_type", pd.Series(["none"])).iloc[0]),
        "final_clean_acc": _last_finite(rounds, "accuracy"),
        "best_clean_acc": float(_number(rounds, "accuracy").max()) if not _number(rounds, "accuracy").dropna().empty else float("nan"),
        "final_basr": _last_finite(rounds, "basr_global"),
        "attack_window_mean_basr": _mean_finite(attack, "basr_global"),
        "attack_window_peak_basr": peak_basr,
        "attack_window_peak_round": peak_round,
        "raw_basr_auc": raw_auc,
        "normalized_basr_auc": normalized_auc,
        "auc_status": auc_status,
        "prototype_observations_final": _last_finite(rounds, "niabd_prototype_observations"),
        "prototype_observations_max": float(prototype.max()) if not prototype.dropna().empty else float("nan"),
        "memory_update_rounds": float(updates.fillna(0).sum()) if "niabd_prototype_updated" in rounds else "不可重建",
        "mean_eligible_teacher_count": _mean_finite(rounds, "niabd_memory_eligible_teachers"),
        "all_ineligible_round_rate": float((eligible == 0).mean()) if not eligible.dropna().empty else "不可重建",
        "longest_freeze_streak": _longest_false(update_flags.tolist()) if "niabd_prototype_updated" in rounds else "不可重建",
        "mean_suppression": _mean_finite(rounds, "niabd_mean_suppression"),
        "admission_rate": float(admission_rate.dropna().mean()) if not admission_rate.dropna().empty else "不可重建",
        "pre_attack_acc_change": "不可重建",
        "version": str(summary.get("niabd_algorithm_version", pd.Series(["不可重建"])).iloc[0]) if "niabd_algorithm_version" in summary else "不可重建",
        "schema": str(summary.get("result_schema_version", pd.Series(["不可重建"])).iloc[0]) if "result_schema_version" in summary else "不可重建",
    }
    attack_start = _last_finite(rounds, "attack_start_round")
    if math.isfinite(attack_start) and "accuracy" in rounds.columns:
        before = _number(rounds.loc[_number(rounds, "round") < attack_start], "accuracy").dropna()
        if not before.empty:
            result["pre_attack_acc_change"] = float(before.iloc[-1] - before.iloc[0])
    if "teachers_admitted" in rounds.columns and "is_malicious" in rounds.columns:
        result["malicious_benign_admission_offline"] = "可重建"
    else:
        result["malicious_benign_admission_offline"] = "不可重建"
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "不可重建"
    if isinstance(value, float) and math.isnan(value):
        return "不可重建"
    return str(value)


def _control_findings(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    niabd = [row for row in rows if row.get("strategy") in {"niabd", "vcaa-niabd"}]
    if niabd:
        stopped = [
            row for row in niabd
            if math.isfinite(float(row.get("prototype_observations_final", float("nan"))))
            and float(row.get("prototype_observations_final"))
            == float(row.get("prototype_observations_max"))
        ]
        findings.append(
            f"NIABD historic rows with observations stopped at final value: {len(stopped)}/{len(niabd)}; exact cause requires versioned v2 fields and is not inferred."
        )
        zero = [row for row in niabd if row.get("all_ineligible_round_rate") == 1.0]
        findings.append(f"NIABD all-ineligible rate exactly 1.0: {len(zero)} rows.")
    attacked = [row for row in rows if row.get("attack_type") != "none" and row.get("attack_window_peak_basr") not in {None, "不可重建"}]
    mid_higher = [
        row for row in attacked
        if isinstance(row.get("attack_window_peak_basr"), (int, float))
        and isinstance(row.get("final_basr"), (int, float))
        and math.isfinite(float(row["attack_window_peak_basr"]))
        and math.isfinite(float(row["final_basr"]))
        and float(row["attack_window_peak_basr"]) > float(row["final_basr"])
    ]
    findings.append(f"Attack-window peak BASR above final BASR: {len(mid_higher)}/{len(attacked)} attacked rows.")
    findings.append("VCAA malicious/benign enrichment: 不可重建 from summary-only historical rows unless raw teacher labels are present.")
    findings.append("Blend attacked baseline versus triggered-no-poison control: compare rows by dataset/attack/seed; no inference is made here when pairing is incomplete.")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit historical raw FedAgg CSV/JSON outputs without modifying them.")
    parser.add_argument("--indir", default="results")
    parser.add_argument("--out", default="HISTORICAL_RESULTS_AUDIT.md")
    parser.add_argument("--machine-output", default="historical_results_audit.json")
    args = parser.parse_args()
    root = Path(args.indir)
    paths = sorted(root.rglob("fedagg_run_summary_*.csv"))
    rows = [_run_metrics(path) for path in paths]
    document = {
        "input_root": str(root),
        "summary_count": len(paths),
        "raw_round_csv_count": len(list(root.rglob("fedagg_experiment_results_*.csv"))),
        "raw_teacher_csv_count": len(list(root.rglob("fedagg_teacher_*.csv"))),
        "raw_runtime_event_csv_count": len(list(root.rglob("fedagg_runtime_events_*.csv"))),
        "json_count": len(list(root.rglob("*.json"))),
        "runs": rows,
        "limitations": [
            "历史文件没有 v3 algorithm/schema/manifest/checkpoint lineage，相关指标标记不可重建。",
            "历史结果未修改；results.zip 不存在，因此没有解压审计副本。",
            "任何防御有效性结论必须重新通过 clean control、triggered-no-poison、攻击可行性和多 seed 门禁。",
        ],
    }
    Path(args.machine_output).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Historical Results Audit",
        "",
        f"审计输入：`{root}`。本次未发现 `results.zip`，因此直接解析原始 round/summary/teacher/runtime CSV 和 JSON；旧文件未修改。",
        "",
        f"- summary CSV：{len(paths)}",
        f"- round CSV：{document['raw_round_csv_count']}",
        f"- teacher CSV：{document['raw_teacher_csv_count']}",
        f"- runtime event CSV：{document['raw_runtime_event_csv_count']}",
        f"- JSON：{document['json_count']}",
        "",
        "## 指标重算规则",
        "",
        "所有指标均直接从原始 CSV 重算：final/best clean ACC、final BASR、attack-window mean/peak/round、raw 与 normalized trapezoidal AUC、prototype observations、memory update rounds、eligible teacher count、all-ineligible rate、freeze streak、suppression 和 admission rate。缺少字段或 round 不连续时写为‘不可重建’，不填估计值。",
        "",
        "## Run-level results",
        "",
        "| dataset | strategy | attack | seed | final ACC | best ACC | final BASR | window mean | window peak | peak round | raw AUC | norm AUC | observations | updates | eligible | all-ineligible | version/schema |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {strategy} | {attack_type} | {seed} | {final_clean_acc} | {best_clean_acc} | {final_basr} | {attack_window_mean_basr} | {attack_window_peak_basr} | {attack_window_peak_round} | {raw_basr_auc} | {normalized_basr_auc} | {prototype_observations_final} | {memory_update_rounds} | {mean_eligible_teacher_count} | {all_ineligible_round_rate} | {version}/{schema} |".format(
                dataset=row.get("dataset", ""), strategy=row.get("strategy", ""), attack_type=row.get("attack_type", ""), seed=row.get("seed", ""),
                final_clean_acc=_fmt(row.get("final_clean_acc")), best_clean_acc=_fmt(row.get("best_clean_acc")), final_basr=_fmt(row.get("final_basr")),
                attack_window_mean_basr=_fmt(row.get("attack_window_mean_basr")), attack_window_peak_basr=_fmt(row.get("attack_window_peak_basr")), attack_window_peak_round=_fmt(row.get("attack_window_peak_round")),
                raw_basr_auc=_fmt(row.get("raw_basr_auc")), normalized_basr_auc=_fmt(row.get("normalized_basr_auc")), prototype_observations_final=_fmt(row.get("prototype_observations_final")),
                memory_update_rounds=_fmt(row.get("memory_update_rounds")), mean_eligible_teacher_count=_fmt(row.get("mean_eligible_teacher_count")), all_ineligible_round_rate=_fmt(row.get("all_ineligible_round_rate")),
                version=_fmt(row.get("version")), schema=_fmt(row.get("schema")),
            )
        )
    lines += ["", "## Confirmed/limited historical phenomena", ""]
    lines.extend(f"- {finding}" for finding in _control_findings(rows))
    lines += [
        "",
        "## Scientific validity boundary",
        "",
        "这些旧结果没有 v3 manifest、完整 checkpoint identity 或 typed schema，因此不能证明 v3 生产链路，也不能将 peak/final 差异解释为 recovery。NIABD 只依赖 clean proxy logits；clean-logit indistinguishability 场景在信息论上不可分，VCAA 只提供时效性/质量准入，不提供后门安全保证。",
        "",
        f"机器可读审计：`{args.machine_output}`。",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[audit] summaries={len(paths)} rounds={document['raw_round_csv_count']} teachers={document['raw_teacher_csv_count']} runtime_events={document['raw_runtime_event_csv_count']}")
    print(f"[write] {args.out}")
    print(f"[write] {args.machine_output}")


if __name__ == "__main__":
    main()
