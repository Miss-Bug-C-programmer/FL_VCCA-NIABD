from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that no file from the recorded upstream main snapshot was "
            "deleted and that critical VCAA/NIABD/server-protocol mechanism "
            "files remain byte-identical."
        )
    )
    parser.add_argument(
        "--manifest",
        default="BASELINE_MAIN_PRESERVATION.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    originals = manifest["original_files_sha256"]
    missing = [rel for rel in originals if not (root / rel).is_file()]
    if missing:
        raise SystemExit(
            "FAIL: original main files are missing:\n" + "\n".join(missing)
        )

    critical = manifest[
        "critical_mechanism_files_must_remain_byte_identical"
    ]
    changed = []
    for rel in critical:
        current = sha256(root / rel)
        expected = originals[rel]
        if current != expected:
            changed.append((rel, expected, current))
    if changed:
        text = "\n".join(
            f"{rel}: expected={expected} current={current}"
            for rel, expected, current in changed
        )
        raise SystemExit(
            "FAIL: critical mechanism files changed relative to source main:\n"
            + text
        )

    print(
        "PASS: all "
        f"{len(originals)} original main files are present; "
        f"all {len(critical)} critical mechanism files are byte-identical "
        f"to {manifest['source_commit']}."
    )


if __name__ == "__main__":
    main()
