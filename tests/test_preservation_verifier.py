import json
import hashlib
import subprocess
import sys
from pathlib import Path


def _baseline(repo: Path) -> dict:
    return json.loads(
        (repo / "BASELINE_MAIN_PRESERVATION.json").read_text(
            encoding="utf-8"
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(
    repo: Path,
    *,
    file: str = "niabd.py",
    patched_sha256: str | None = None,
) -> dict:
    baseline = _baseline(repo)
    return {
        "file": file,
        "upstream_sha256": baseline["original_files_sha256"][file],
        "patched_sha256": patched_sha256 or "f" * 64,
        "reason": "test authorization",
        "reproducer": "test command",
        "fix_report": "test report",
        "tests": ["test_preservation_verifier.py"],
    }


def _run(repo: Path, authorization: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_preserved_main.py"),
            "--root",
            str(repo),
            "--authorizations",
            str(authorization),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def test_preservation_fails_closed_without_authorization(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    authorization = tmp_path / "missing.json"
    result = _run(repo, authorization)
    assert result.returncode != 0
    assert "unauthorized baseline file changes" in result.stderr + result.stdout


def test_preservation_rejects_wildcard_authorization(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    entry = _entry(repo)
    entry["file"] = "*.py"
    document = {
        "schema_version": "fedagg-authorized-core-fixes-v1",
        "authorized_files": [entry],
    }
    authorization = tmp_path / "wildcard.json"
    authorization.write_text(json.dumps(document), encoding="utf-8")
    result = _run(repo, authorization)
    assert result.returncode != 0
    assert "wildcard" in result.stderr + result.stdout


def test_preservation_rejects_patched_hash_mismatch(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    document = {
        "schema_version": "fedagg-authorized-core-fixes-v1",
        "authorized_files": [
            _entry(
                repo,
                file="defense.py",
                patched_sha256=_sha256(repo / "defense.py"),
            ),
            _entry(repo),
        ],
    }
    authorization = tmp_path / "mismatch.json"
    authorization.write_text(json.dumps(document), encoding="utf-8")
    result = _run(repo, authorization)
    assert result.returncode != 0
    assert "patched hashes" in result.stderr + result.stdout
