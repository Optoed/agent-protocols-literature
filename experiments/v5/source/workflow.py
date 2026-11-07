"""Validated acyclic agent workflows over the existing FEDOT runtime."""

from __future__ import annotations

import hashlib
import json
from typing import Self

from fedotmas import blackboard
from fedotmas.engine import Store
from fedotmas_llm import LLM, Call, PromptRule
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class InvalidProposal(ValueError):
    """The generator exhausted its explicit graph-repair attempts."""


class WorkflowNode(BaseModel):
    """One model call with explicitly named input facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    prompt: str = Field(min_length=1)
    inputs: tuple[str, ...] = Field(min_length=1)


class WorkflowSpec(BaseModel):
    """An ordered DAG; every node contributes to the final answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[WorkflowNode, ...] = Field(min_length=1, max_length=6)
    output: str

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        available = {"question"}
        ancestors: dict[str, set[str]] = {"question": set()}
        for node in self.nodes:
            if node.name in available:
                raise ValueError(f"duplicate or reserved name: {node.name}")
            if len(set(node.inputs)) != len(node.inputs):
                raise ValueError(f"duplicate inputs: {node.name}")
            if not set(node.inputs) <= available:
                raise ValueError(f"unknown or forward input: {node.name}")
            ancestors[node.name] = set(node.inputs).union(
                *(ancestors[name] for name in node.inputs)
            )
            available.add(node.name)
        names = {node.name for node in self.nodes}
        if self.output not in names:
            raise ValueError("output must name an agent node")
        if ancestors[self.output] | {self.output} != available:
            raise ValueError("every node must contribute to the output")
        return self

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def topology(self) -> tuple[tuple[int, ...], ...]:
        positions = {"question": -1} | {
            node.name: index for index, node in enumerate(self.nodes)
        }
        return tuple(
            tuple(sorted(positions[x] for x in node.inputs)) for node in self.nodes
        )


async def execute_workflow(spec: WorkflowSpec, question: str, llm: LLM) -> str:
    """Execute the DAG once, with no monetary cap inside the workflow."""
    board = blackboard(
        *(
            PromptRule(
                name=node.name,
                writes=node.name,
                when=[*node.inputs, f"!{node.name}"],
                prompt=node.prompt,
                input="\n\n".join(f"{name}:\n{{{name}}}" for name in node.inputs),
            )
            for node in spec.nodes
        )
    )
    result = await board.run(
        {"question": question}, goal=spec.output, bind={"llm": llm}, budget=None
    )
    value = result.unwrap()
    if not isinstance(value, str):
        raise TypeError("workflow output must be text")
    return value


async def propose_workflow(
    parent: WorkflowSpec, feedback: str, llm: LLM, *, repairs: int = 1
) -> WorkflowSpec:
    """Propose a structural mutation, repairing invalid proposals explicitly."""
    if repairs < 0:
        raise ValueError("repairs must be nonnegative")
    prompt = (
        "Improve an agent workflow by changing its structure: insert/remove a review, "
        "revision, independent solution, or aggregation node; or change input edges. "
        "Return a WorkflowSpec with 1 to 6 nodes in topological order. Inputs reference "
        "'question' or earlier node names. Every node must contribute to output. "
        "Use lowercase identifiers. Prompts are system instructions for each agent. "
        "The final node must return the answer in \\boxed{...}. Do not put benchmark "
        "answers or task-specific constants into prompts. Changing only prompts or "
        "renaming nodes is insufficient. Return only the requested structured object."
    )
    request = {"parent": parent.model_dump(mode="json"), "feedback": feedback}
    for attempt in range(repairs + 1):
        try:
            proposed = await llm.complete(
                Call(prompt, request, returns=WorkflowSpec), Store().snapshot()
            )
            child = WorkflowSpec.model_validate(proposed)
            if child.topology == parent.topology:
                raise InvalidProposal(
                    "proposal must change structure, not just prompts"
                )
            return child
        except (ValidationError, InvalidProposal) as exc:
            if attempt == repairs:
                raise InvalidProposal(str(exc)) from exc
            request["validation_error"] = str(exc)
    raise AssertionError("unreachable")
