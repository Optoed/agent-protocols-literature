"""Package the original V5 evidence and complete tracked execution source."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    destination = Path(__file__).resolve().parent
    workspace = destination.parents[3]
    source = workspace / "swarm-dev"
    runs = workspace / "workflow-runs"
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=source
    ).decode().split("\0")
    source_files = {
        f"runtime-source/{name}": (source / name).read_bytes()
        for name in tracked
        if name and (source / name).is_file()
    }
    evidence = {}
    chain = ["main-v5-confirmed", "main-v5-confirmed-recovery1", "main-v5-confirmed-recovery2"]
    folders = [runs / name for name in chain]
    folders += [runs / "math-algebra-data-v1", runs / "math-intermediate-level5-v1"]
    folders += [p for p in (runs / "commands").iterdir() if "v5" in p.name]
    audit = workspace / "protocol-selection-review"
    audit_folder = audit / "qa/audit-20261105"
    folders += [audit_folder / "offline-replay"]
    folders += list(audit_folder.glob("v5*.json"))
    folders += list(audit_folder.glob("hf-*.json"))
    folders += [audit_folder / "dataset-live-checks.json"]
    for folder in folders:
        paths = folder.rglob("*") if folder.is_dir() else [folder]
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                evidence[path.relative_to(workspace).as_posix()] = path.read_bytes()
    for path in runs.glob("*.json"):
        if "plan" in path.name or path.name in {
            "main-comparison-v1.json", "credit-main-preflight.json",
            "credit-v3-preflight.json", "credit-recovery2-preflight.json",
        }:
            evidence[path.relative_to(workspace).as_posix()] = path.read_bytes()
    for path in (runs / "diagnostics").glob("v5*.json"):
        evidence[path.relative_to(workspace).as_posix()] = path.read_bytes()
    for name in ("audit_v5_20261105.py", "audit_independent_scoring_20261105.py", "audit_remote_provenance_20261105.py"):
        path = audit / "tools" / name
        evidence[path.relative_to(workspace).as_posix()] = path.read_bytes()
    evidence["protocol-selection-review/EXPERIMENT_AUDIT_V5_2026-11-05.md"] = (
        audit / "EXPERIMENT_AUDIT_V5_2026-11-05.md"
    ).read_bytes()
    forbidden = re.compile(rb"sk-or-v1-[0-9a-fA-F]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    for name, content in (source_files | evidence).items():
        if forbidden.search(content) or Path(name).name == ".env":
            raise ValueError(f"potential credential in {name}")
    manifests = {}
    for name, contents in [("runtime-source.zip", source_files), ("evidence.zip", evidence)]:
        with ZipFile(destination / name, "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, content in sorted(contents.items()):
                archive.writestr(relative, content)
            manifest = {relative: digest(content) for relative, content in sorted(contents.items())}
            archive.writestr("SHA256.json", json.dumps(manifest, indent=2))
        with ZipFile(destination / name) as archive:
            assert archive.testzip() is None
            for relative, expected in manifest.items():
                assert digest(archive.read(relative)) == expected
        manifests[name] = {
            "sha256": digest((destination / name).read_bytes()),
            "bytes": (destination / name).stat().st_size,
            "files": len(contents),
        }
    protocol = json.loads((runs / chain[-1] / "protocol.json").read_text())
    for name, expected in protocol["source_hashes"].items():
        key = "runtime-source/" + name.replace("\\", "/")
        assert digest(source_files[key]) == expected, name
    from importlib.metadata import distributions

    manifest = {
        "created_on": "2026-11-09",
        "author": "Petr Stekliannikov",
        "experiment_commit": "b0bcaea169794e0feb04918ebcb59934dda9e914",
        "execution_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source).decode().strip(),
        "chain": chain,
        "accounting": "Use only the final recovery directory, including interrupted_attempts/calls. Do not sum the three inherited copies.",
        "archives": manifests,
        "frozen_source_hashes_match": True,
        "python": sys.version,
        "installed_distributions": sorted({(d.metadata["Name"], d.version) for d in distributions()}),
        "secrets": "No environment files, API keys or virtual environment included. Known credential patterns scanned before packaging.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifests))


if __name__ == "__main__":
    main()
