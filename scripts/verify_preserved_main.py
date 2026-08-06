from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict


_AUTH_REQUIRED_KEYS = {
    "file",
    "upstream_sha256",
    "patched_sha256",
    "reason",
    "reproducer",
    "fix_report",
    "tests",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head_sha256(root: Path, relative: str) -> str:
    """Hash a file tracked by the pre-fix repository HEAD."""

    try:
        content = subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{relative}"],
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"FAIL: cannot read pre-fix HEAD blob for authorized file: {relative}"
        ) from exc
    return hashlib.sha256(content).hexdigest()


def _load_authorizations(
    *,
    root: Path,
    manifest: dict,
    path: Path,
) -> Dict[str, dict]:
    """Load exact file-level hash transitions for reviewed core fixes."""

    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit("FAIL: authorization manifest must be a JSON object.")
    if document.get("schema_version") != "fedagg-authorized-core-fixes-v1":
        raise SystemExit("FAIL: invalid authorization schema_version.")
    entries = document.get("authorized_files")
    if not isinstance(entries, list):
        raise SystemExit("FAIL: authorized_files must be a list.")

    originals = manifest["original_files_sha256"]
    result: Dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _AUTH_REQUIRED_KEYS:
            raise SystemExit(
                "FAIL: every authorization entry must contain exactly "
                f"{sorted(_AUTH_REQUIRED_KEYS)}."
            )
        relative = entry["file"]
        if not isinstance(relative, str) or not relative:
            raise SystemExit("FAIL: authorization file must be a non-empty string.")
        if any(character in relative for character in "*?[]"):
            raise SystemExit(f"FAIL: authorization uses a wildcard: {relative}")
        candidate = Path(relative)
        if candidate.is_absolute() or relative.endswith(("/", "\\")):
            raise SystemExit(
                f"FAIL: authorization must target one relative file: {relative}"
            )
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise SystemExit(
                f"FAIL: authorization escapes repository root: {relative}"
            ) from exc
        normalized = candidate.as_posix()
        if normalized not in originals:
            expected_upstream = _git_head_sha256(root, normalized)
        else:
            expected_upstream = originals[normalized]
        if normalized in result:
            raise SystemExit(f"FAIL: duplicate authorization: {relative}")
        if not resolved.is_file() or resolved.is_dir():
            raise SystemExit(f"FAIL: authorized file is missing: {relative}")
        for key in ("upstream_sha256", "patched_sha256"):
            value = entry[key]
            if not isinstance(value, str) or len(value) != 64:
                raise SystemExit(
                    f"FAIL: {relative} has invalid {key}; expected SHA-256."
                )
        if entry["upstream_sha256"] != expected_upstream:
            raise SystemExit(
                f"FAIL: {relative} upstream hash does not match its recorded source."
            )
        if entry["patched_sha256"] == entry["upstream_sha256"]:
            raise SystemExit(
                f"FAIL: {relative} authorization does not describe a change."
            )
        for key in ("reason", "reproducer", "fix_report"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise SystemExit(
                    f"FAIL: {relative} authorization is missing {key}."
                )
        tests = entry["tests"]
        if (
            not isinstance(tests, list)
            or not tests
            or any(
                not isinstance(item, str) or not item.strip()
                for item in tests
            )
        ):
            raise SystemExit(
                f"FAIL: {relative} authorization must list tests."
            )
        result[normalized] = entry
    return result


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
    parser.add_argument(
        "--authorizations",
        default="AUTHORIZED_CORE_FIXES.json",
        help="Exact per-file hash transitions for reviewed production fixes.",
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

    authorization_path = Path(args.authorizations)
    if not authorization_path.is_absolute():
        authorization_path = root / authorization_path
    authorizations = _load_authorizations(
        root=root,
        manifest=manifest,
        path=authorization_path,
    )

    critical = manifest[
        "critical_mechanism_files_must_remain_byte_identical"
    ]
    unauthorized = []
    mismatched_authorizations = []
    for rel in critical:
        expected = originals[rel]
        current = sha256(root / rel)
        if current == expected:
            continue
        authorization = authorizations.get(rel)
        if authorization is None:
            unauthorized.append((rel, expected, current))
        elif current != authorization["patched_sha256"]:
            mismatched_authorizations.append(
                (rel, authorization["patched_sha256"], current)
            )
    if unauthorized:
        text = "\n".join(
            f"{rel}: expected={expected} current={current}"
            for rel, expected, current in unauthorized
        )
        raise SystemExit("FAIL: unauthorized baseline file changes:\n" + text)
    if mismatched_authorizations:
        text = "\n".join(
            f"{rel}: authorized={expected} current={current}"
            for rel, expected, current in mismatched_authorizations
        )
        raise SystemExit(
            "FAIL: authorized patched hashes do not match:\n" + text
        )

    for rel, authorization in authorizations.items():
        current = sha256(root / rel)
        if current != authorization["patched_sha256"]:
            raise SystemExit(
                "FAIL: authorized patched hashes do not match:\n"
                f"{rel}: authorized={authorization['patched_sha256']} "
                f"current={current}"
            )

    print(
        "PASS: all "
        f"{len(originals)} original main files are present; "
        f"all critical changes are covered by {len(authorizations)} exact file-level "
        f"authorization(s); baseline={manifest['source_commit']}."
    )


if __name__ == "__main__":
    main()
