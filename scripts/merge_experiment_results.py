"""Union compatible FedAgg result tables from old and supplementary roots."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


TABLE_NAMES = (
    "fedagg_run_summary_",
    "fedagg_experiment_results_",
    "fedagg_backdoor_defense_",
    "fedagg_teacher_admission_",
    "fedagg_teacher_defense_",
    "fedagg_runtime_events_",
)
KNOWN_STRATEGIES = {"baseline", "vcaa", "niabd", "vcaa-niabd"}
KNOWN_ATTACKS = {"none", "badnets", "dba", "blend", "dynamic"}


def _table_kind(path: Path) -> str | None:
    name = path.name
    for prefix in TABLE_NAMES:
        if name.startswith(prefix) and name.endswith(".csv"):
            return prefix
    return None


def _key_columns(kind: str, frame: pd.DataFrame) -> list[str]:
    base = ["dataset", "attack_type", "strategy", "seed"]
    if kind != "fedagg_run_summary_":
        base.append("round")
    if kind in {"fedagg_teacher_admission_", "fedagg_teacher_defense_", "fedagg_backdoor_defense_"}:
        base.append("client_id")
    if kind == "fedagg_runtime_events_":
        base.extend(column for column in ("packet_id", "task_id") if column in frame.columns)
    missing = [column for column in base if column not in frame.columns]
    if missing and not frame.empty:
        raise ValueError(
            f"cannot check duplicate keys for {kind}: missing {missing}"
        )
    return base + [
        column for column in (
            "round", "client_id", "packet_id", "task_id",
        )
        if column in frame.columns and column not in base
    ] if not missing else []


def _path_identity(path: Path, column: str):
    """Infer only a duplicate-check key from the v3 directory identity.

    Older teacher-level CSVs predate strategy/attack columns.  The inferred
    values are kept out of the output frame; they only make duplicate checks
    possible without rewriting immutable historical data.
    """

    parts = list(path.parts)
    if column == "strategy":
        for part in reversed(parts):
            if part in KNOWN_STRATEGIES:
                return part
    if column == "attack_type":
        for part in reversed(parts):
            if part in KNOWN_ATTACKS:
                return part
    if column == "seed":
        for part in reversed(parts):
            match = re.fullmatch(r"seed_(\d+)", part)
            if match:
                return int(match.group(1))
    if column == "dataset":
        stem = path.name
        for prefix in TABLE_NAMES:
            if stem.startswith(prefix) and stem.endswith(".csv"):
                return stem[len(prefix):-4]
    return None


def _files(roots: Iterable[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"result root does not exist: {root}")
        for path in sorted(root.rglob("*.csv")):
            kind = _table_kind(path)
            if kind is not None:
                grouped.setdefault(path.name, []).append(path)
    return grouped


def merge(roots: list[str], outdir: str, *, precedence: str = "error") -> dict[str, int]:
    if precedence not in {"error", "newer", "older"}:
        raise ValueError("precedence must be error, newer, or older")
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for filename, paths in _files([Path(root) for root in roots]).items():
        frames = []
        key_frames = []
        kind = _table_kind(Path(filename))
        assert kind is not None
        columns: list[str] = []
        for path in paths:
            frame = pd.read_csv(path)
            frames.append(frame)
            key_frame = frame.copy()
            for identity_column in ("dataset", "attack_type", "strategy", "seed"):
                if identity_column not in key_frame.columns:
                    inferred = _path_identity(path, identity_column)
                    if inferred is not None:
                        key_frame[identity_column] = inferred
            key_frames.append(key_frame)
            for column in frame.columns:
                if column not in columns:
                    columns.append(column)
        if not frames:
            continue
        # Header-only strategy tables are expected.  Concatenating only the
        # non-empty frames avoids pandas dtype warnings while the explicit
        # column union below still preserves every historical/new header.
        data_frames = [frame for frame in frames if not frame.empty] or frames[:1]
        data_key_frames = [
            frame for frame in key_frames if not frame.empty
        ] or key_frames[:1]
        merged = pd.concat(
            data_frames,
            ignore_index=True,
            sort=False,
        ).reindex(columns=columns)
        merged_keys = pd.concat(
            data_key_frames,
            ignore_index=True,
            sort=False,
        )
        key = _key_columns(kind, merged_keys)
        if key:
            duplicate = merged_keys.duplicated(key, keep=False)
            if bool(duplicate.any()):
                if precedence == "error":
                    examples = merged_keys.loc[duplicate, key].head(5).to_dict("records")
                    raise ValueError(f"duplicate result keys in {filename}: {examples}")
                keep = "last" if precedence == "newer" else "first"
                keep_mask = ~merged_keys.duplicated(key, keep=keep)
                merged = merged.loc[keep_mask].reset_index(drop=True)
        merged.to_csv(target / filename, index=False)
        summary[filename] = int(len(merged))
    if not summary:
        raise FileNotFoundError("No supported FedAgg result CSV files found.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--precedence", choices=["error", "newer", "older"], default="error")
    args = parser.parse_args()
    for filename, count in merge(args.roots, args.outdir, precedence=args.precedence).items():
        print(f"{filename}: {count} rows")


if __name__ == "__main__":
    main()
