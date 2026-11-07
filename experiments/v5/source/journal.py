"""Durable provider calls with explicit unknown costs and replayable responses."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fedotmas.engine.contract import View
from fedotmas_llm import Call
from pydantic import BaseModel


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def archive_directory(root: Path, name: str) -> Path:
    if not root.is_absolute() or not Path(root.anchor).exists():
        raise FileNotFoundError("archive drive is unavailable; no paid calls started")
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("run name must be one directory name")
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class Journal:
    def __init__(self, directory: Path, client: httpx.AsyncClient):
        self.directory = directory
        self.client = client

    async def request(self, identity: str, body: dict, category: str) -> dict:
        key = hashlib.sha256(identity.encode()).hexdigest()
        path = self.directory / "calls" / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["request"] != body or record["category"] != category:
                raise ValueError("resume configuration differs from recorded request")
            if record["status"] != "completed":
                raise RuntimeError(
                    "unsettled provider call; inspect before retrying payment"
                )
            return record["response"]
        record = {
            "identity": identity,
            "request": body,
            "category": category,
            "status": "pending",
            "cost": None,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(path, record)
        started = time.monotonic()
        try:
            response = await self.client.post(
                "https://openrouter.ai/api/v1/chat/completions", json=body
            )
            response.raise_for_status()
            data = response.json()
            cost = data.get("usage", {}).get("cost")
            if cost is not None and (
                not isinstance(cost, (int, float))
                or not math.isfinite(cost)
                or cost < 0
            ):
                raise ValueError("provider returned invalid cost")
            record.update(
                status="completed",
                response=data,
                cost=cost,
                elapsed_seconds=time.monotonic() - started,
            )
            write_json(path, record)
            return data
        except Exception as exc:
            record.update(
                status="unsettled",
                error_type=type(exc).__name__,
                elapsed_seconds=time.monotonic() - started,
            )
            write_json(path, record)
            raise

    def costs(self) -> dict:
        result: dict[str, dict] = {}
        for path in (self.directory / "calls").glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            group = result.setdefault(
                record["category"],
                {
                    "calls": 0,
                    "known_usd": 0.0,
                    "unknown_cost_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "unknown_usage_calls": 0,
                },
            )
            group["calls"] += 1
            usage = record.get("response", {}).get("usage", {})
            if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                group["unknown_usage_calls"] += 1
            else:
                group["prompt_tokens"] += usage["prompt_tokens"]
                group["completion_tokens"] += usage["completion_tokens"]
            if record.get("cost") is None:
                group["unknown_cost_calls"] += 1
            else:
                group["known_usd"] += record["cost"]
        return result


class JournalLLM:
    def __init__(self, journal: Journal, model: str, scope: str, category: str):
        self.journal, self.model = journal, model
        self.scope, self.category = scope, category
        self.sequence = 0

    async def complete(self, call: Call, view: View) -> Any:
        if call.tools:
            raise ValueError("tools are outside the first workflow-search experiment")
        structured = isinstance(call.returns, type) and issubclass(
            call.returns, BaseModel
        )
        prompt = call.prompt
        if structured:
            prompt += "\nReturn JSON matching this schema:\n" + json.dumps(
                call.returns.model_json_schema()
            )
        body = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(call.input, ensure_ascii=False)},
            ],
        }
        if structured:
            body["response_format"] = {"type": "json_object"}
        identity = f"{self.scope}/call-{self.sequence}"
        self.sequence += 1
        data = await self.journal.request(identity, body, self.category)
        choice = data["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(
                "incomplete provider answer; charged response is preserved"
            )
        content = choice["message"]["content"]
        return call.returns.model_validate_json(content) if structured else content
