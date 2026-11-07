"""Reuse identical initial calls across workflows while keeping independent draws distinct."""

import asyncio
import hashlib
import json
from collections import Counter
from typing import Any

from fedotmas.engine.contract import View
from fedotmas_llm import Call

from benchmarks.workflow_search.journal import Journal, JournalLLM


class SharedRoots:
    def __init__(
        self, journal: Journal, model: str, scope: str, question: str, category: str
    ):
        self.journal, self.model, self.scope = journal, model, scope
        self.root_input = f"question:\n{question}"
        self.category = category
        self.pending: dict[str, asyncio.Task] = {}

    def workflow(self, fingerprint: str) -> "SharedRootLLM":
        return SharedRootLLM(self, f"{self.scope}/workflow/{fingerprint}")


class SharedRootLLM(JournalLLM):
    def __init__(self, pool: SharedRoots, scope: str):
        super().__init__(pool.journal, pool.model, scope, pool.category)
        self.pool = pool
        self.occurrences: Counter[str] = Counter()
        self.record_scopes = {scope}

    async def complete(self, call: Call, view: View) -> Any:
        if call.input != self.pool.root_input or call.tools or call.returns is not str:
            return await super().complete(call, view)
        digest = hashlib.sha256(
            json.dumps(
                [self.model, call.prompt, call.input], ensure_ascii=False
            ).encode()
        ).hexdigest()
        ordinal = self.occurrences[digest]
        self.occurrences[digest] += 1
        scope = f"{self.pool.scope}/root/{digest}/draw-{ordinal}"
        self.record_scopes.add(scope)
        if scope not in self.pool.pending:
            backend = JournalLLM(self.journal, self.model, scope, self.category)
            self.pool.pending[scope] = asyncio.create_task(backend.complete(call, view))
        return await self.pool.pending[scope]
