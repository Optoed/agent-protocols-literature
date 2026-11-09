"""Run the complete frozen V5 selection procedure on saved API responses, offline."""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    for package in ("fedotmas", "fedotmas-llm", "fedotmas-meta"):
        sys.path.insert(0, str(source / "packages" / package / "src"))
    sys.path.insert(0, str(source))
    from benchmarks.workflow_search.confirmed import experiment
    from benchmarks.workflow_search.journal import write_json

    run = args.evidence / "workflow-runs/main-v5-confirmed-recovery2"
    plan_path = args.evidence / "workflow-runs/main-v5-confirmed-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    stored = json.loads((run / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == protocol["plan_sha256"]
    for relative, expected in protocol["source_hashes"].items():
        assert hashlib.sha256((source / relative.replace("\\", "/")).read_bytes()).hexdigest() == expected
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (run / "calls").glob("*.json")]
    assert all(r["status"] == "completed" for r in records)
    by_identity = {r["identity"]: r for r in records}
    assert len(by_identity) == len(records)

    class OfflineJournal:
        directory = run

        def __init__(self):
            self.used = set()

        async def request(self, identity, body, category):
            record = by_identity[identity]
            assert record["request"] == body and record["category"] == category
            assert identity not in self.used
            self.used.add(identity)
            return record["response"]

        def costs(self):
            return stored["costs"]

    args.output.mkdir(parents=True, exist_ok=False)
    journal = OfflineJournal()
    result = asyncio.run(experiment(plan, args.output, journal))
    assert not result["errors"] and result["replicates"] == stored["replicates"]
    assert len(journal.used) == len(records) == 3312
    evaluations = list((run / "evaluations").glob("*.json"))
    for path in evaluations:
        assert json.loads(path.read_text(encoding="utf-8")) == json.loads((args.output / "evaluations" / path.name).read_text(encoding="utf-8"))
    write_json(args.output / "replay-check.json", {"status": "pass", "calls": len(records), "evaluations": len(evaluations), "network_requests": 0})
    print(json.dumps({"status": "pass", "calls": len(records), "evaluations": len(evaluations), "network_requests": 0}))


if __name__ == "__main__":
    main()
