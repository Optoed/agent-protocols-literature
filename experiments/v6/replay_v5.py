"""Measure exact-pruning call savings on the historical V5 confirmation table."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    sys.path.insert(0, str(root / "swarm-dev"))
    from benchmarks.workflow_search.progressive import policy_decision
    from fedotmas_meta.workflow import WorkflowSpec

    plan = json.loads((root / "workflow-runs/main-v5-confirmed-plan.json").read_text(encoding="utf-8"))
    run = root / "workflow-runs/main-v5-confirmed-recovery2"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    specs = {fp: WorkflowSpec.model_validate(value) for fp, value in plan["workflows"].items()}
    calls = [json.loads(p.read_text(encoding="utf-8")) for p in (run / "calls").glob("*.json")]
    rows = []
    for rep in result["replicates"]:
        index = rep["replicate"]
        confirmation = {
            fp: [json.loads((run / "evaluations" / f"{index}-confirmation_search-{task['id']}-{fp}.json").read_text(encoding="utf-8")) for task in plan["confirmation"]]
            for fp in rep["finalists"]
        }
        decision = policy_decision("race8", rep["scores"]["shared_search"], confirmation, specs)
        assert decision["winner"] == rep["winners"]["shared_confirm"]
        prefixes = [f"confirmation_search/rep-{index}/{plan['confirmation'][i]['id']}/{fp}/" for i, fp in decision["consumed"]]
        consumed = [r for r in calls if any(r["identity"].startswith(prefix) for prefix in prefixes)]
        assert all(r["status"] == "completed" and r["cost"] is not None for r in consumed)
        rows.append({"replicate": index, "winner_unchanged": True, "winner": decision["winner"], "consumed_workflow_tasks": len(decision["consumed"]), "confirmation_calls": len(consumed), "known_confirmation_usd": sum(r["cost"] for r in consumed), "test_correct_unchanged": rep["test"]["shared_confirm"]["correct"]})
    shared = result["costs"]["shared_search"]["known_usd"]
    baseline = shared + result["costs"]["confirmation_search"]["known_usd"]
    projected = shared + sum(row["known_confirmation_usd"] for row in rows)
    output = {
        "kind": "historical_offline_counterfactual_not_a_new_LLM_experiment",
        "policy": "race8", "replicates": rows,
        "baseline_known_search_usd": baseline,
        "projected_known_search_usd": projected,
        "projected_saving_vs_v5": 1 - projected / baseline,
        "projected_saving_vs_full": 1 - projected / result["costs"]["independent_search"]["known_usd"],
        "new_api_calls": 0,
        "limitations": "Uses historical V5 answer table, including previously seen test outcomes. Demonstrates accounting and equal winners for a fixed table, not fresh performance or measured runtime savings. Original 13 unknown attempts remain unresolved; they are not made free by this calculation.",
    }
    path = Path(__file__).with_name("v5-replay.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
