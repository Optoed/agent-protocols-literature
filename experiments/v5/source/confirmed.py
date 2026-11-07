"""Compare full evaluation with shared screening and independent finalist confirmation."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values
from fedotmas_meta.workflow import WorkflowSpec

from benchmarks.workflow_search.answers import grade as strict_grade
from benchmarks.workflow_search.answers_v2 import grade
from benchmarks.workflow_search.journal import JournalLLM, archive_directory, write_json
from benchmarks.workflow_search.recover import execute_with_length_failure
from benchmarks.workflow_search.recover_paired import TransportRecoveryJournal
from benchmarks.workflow_search.run import validate_tasks
from benchmarks.workflow_search.shared_roots import SharedRoots


def validate_plan(plan: dict):
    validate_tasks(
        {"search": plan["search"] + plan["confirmation"], "test": plan["test"]}
    )
    if not plan["search"] or not plan["confirmation"] or plan["replicates"] < 1:
        raise ValueError("search, confirmation tasks and replicates required")
    if (
        plan["max_tokens"] != 8192
        or plan["temperature"] != 0.2
        or plan["near_tie_margin"] != 1
    ):
        raise ValueError("unsupported frozen settings")
    specs = {k: WorkflowSpec.model_validate(v) for k, v in plan["workflows"].items()}
    if len(specs) < 2:
        raise ValueError("at least two workflows required")
    if any(k != s.fingerprint for k, s in specs.items()):
        raise ValueError("workflow fingerprint mismatch")


def select(scores: dict, specs: dict) -> str:
    return min(scores, key=lambda fp: (-scores[fp], len(specs[fp].nodes), fp))


def finalists(scores: dict, margin: int) -> list[str]:
    best = max(scores.values())
    return [fp for fp, value in scores.items() if best - value <= margin]


async def experiment(
    plan: dict, directory: Path, journal: TransportRecoveryJournal
) -> dict:
    validate_plan(plan)
    specs = {k: WorkflowSpec.model_validate(v) for k, v in plan["workflows"].items()}

    async def evaluate(rep, phase, tasks, keys, shared=False):
        outcomes = {fp: [] for fp in keys}
        for task in tasks:
            scope = f"{phase}/rep-{rep}/{task['id']}"
            pool = SharedRoots(journal, plan["model"], scope, task["question"], phase)
            for fp in keys:
                llm = (
                    pool.workflow(fp)
                    if shared
                    else JournalLLM(journal, plan["model"], f"{scope}/{fp}", phase)
                )
                answer = await execute_with_length_failure(
                    specs[fp], task["question"], llm
                )
                quality, status = grade(answer, task["answer"])
                row = {
                    "id": task["id"],
                    "domain": task.get("domain", "unspecified"),
                    "quality": quality,
                    "parse_status": status,
                    "strict_quality": strict_grade(answer, task["answer"])[0],
                    "answer": answer,
                }
                outcomes[fp].append(row)
                write_json(
                    directory / "evaluations" / f"{rep}-{phase}-{task['id']}-{fp}.json",
                    row,
                )
                if phase == "test_execution":
                    write_json(directory / f"test-{rep}-{fp}.json", outcomes[fp])
            print(
                json.dumps(
                    {
                        "replicate": rep,
                        "phase": phase,
                        "tasks_completed": len(next(iter(outcomes.values()))),
                        "total_tasks": len(tasks),
                        "workflows": len(keys),
                    }
                ),
                flush=True,
            )
        return {
            fp: sum(row["quality"] for row in rows) for fp, rows in outcomes.items()
        }, outcomes

    async def replicate(rep):
        scores = {}
        for phase in (
            ("independent_search", "shared_search")
            if rep % 2 == 0
            else ("shared_search", "independent_search")
        ):
            tasks = (
                plan["search"] + plan["confirmation"]
                if phase == "independent_search"
                else plan["search"]
            )
            scores[phase], _ = await evaluate(
                rep, phase, tasks, list(specs), phase == "shared_search"
            )
        chosen = finalists(scores["shared_search"], plan["near_tie_margin"])
        write_json(
            directory / f"screening-{rep}.json",
            {
                "scores": scores,
                "finalists": chosen,
                "confirmation_needed": len(chosen) > 1,
            },
        )
        confirmation = {}
        combined = {fp: scores["shared_search"][fp] for fp in chosen}
        if len(chosen) > 1:
            confirmation, _ = await evaluate(
                rep, "confirmation_search", plan["confirmation"], chosen
            )
            combined = {
                fp: scores["shared_search"][fp] + confirmation[fp] for fp in chosen
            }
        winners = {
            "independent": select(scores["independent_search"], specs),
            "shared": select(scores["shared_search"], specs),
            "shared_confirm": select(combined, specs),
        }
        selection = {
            "replicate": rep,
            "scores": scores,
            "finalists": chosen,
            "confirmation": confirmation,
            "combined": combined,
            "winners": winners,
        }
        write_json(directory / f"selection-{rep}.json", selection)
        _, rows = await evaluate(
            rep, "test_execution", plan["test"], sorted(set(winners.values()))
        )
        tests = {}
        for mode, fp in winners.items():
            answers = rows[fp]
            tests[mode] = {
                "workflow": fp,
                "correct": sum(row["quality"] for row in answers),
                "count": len(answers),
                "quality": sum(row["quality"] for row in answers) / len(answers),
                "by_domain": {
                    domain: {
                        "correct": sum(
                            row["quality"] for row in answers if row["domain"] == domain
                        ),
                        "count": sum(row["domain"] == domain for row in answers),
                    }
                    for domain in sorted({row["domain"] for row in answers})
                },
            }
        result = selection | {"test": tests}
        write_json(directory / f"replicate-{rep}.json", result)
        return result

    results = await asyncio.gather(
        *(replicate(rep) for rep in range(plan["replicates"])), return_exceptions=True
    )
    errors = [
        {"replicate": i, "type": type(r).__name__}
        for i, r in enumerate(results)
        if isinstance(r, BaseException)
    ]
    return {
        "status": "interrupted" if errors else "completed",
        "errors": errors,
        "replicates": [r for r in results if not isinstance(r, BaseException)],
        "costs": journal.costs(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    directory = archive_directory(args.plan.resolve().parent, args.name)
    root = Path(__file__).resolve().parents[2]
    sources = [
        *Path(__file__).parent.glob("*.py"),
        root / "packages/fedotmas-meta/src/fedotmas_meta/workflow.py",
    ]
    protocol = {
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "transport_retries": 2,
    }
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        parser.error("protocol changed")
    key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(args.env_file).get(
        "OPENROUTER_API_KEY"
    )
    if not key:
        parser.error("missing API key")
    lock = directory / "active.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    async def run():
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(300, connect=30),
        ) as client:
            journal = TransportRecoveryJournal(directory, client, 2)
            try:
                result = await experiment(plan, directory, journal)
                write_json(directory / "result.json", result)
                print(json.dumps(result), flush=True)
                if result["errors"]:
                    raise RuntimeError("experiment interrupted; inspect journal")
            finally:
                write_json(directory / "costs.json", journal.costs())

    try:
        write_json(path, protocol)
        asyncio.run(run())
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
