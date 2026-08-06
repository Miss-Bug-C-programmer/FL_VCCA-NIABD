"""Fail-closed validation for v3 result tables and run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_schema import RESULT_SCHEMA_VERSION, validate_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    args = parser.parse_args()
    root = Path(args.indir).resolve()
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("FAIL: run_manifest.json is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if recorded != actual:
        raise SystemExit("FAIL: run manifest hash mismatch.")
    checked = 0
    for path in sorted(root.glob("fedagg_*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        validate_frame(frame, require_rounds=("summary" not in path.name))
        checked += 1
    if checked == 0:
        raise SystemExit("FAIL: no non-empty v3 result table found.")
    print(f"PASS: {checked} v3 result table(s), schema={RESULT_SCHEMA_VERSION}.")


if __name__ == "__main__":
    main()
