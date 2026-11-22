"""LLM backends: pluggable interface plus deterministic mock backend. Mock parses goal into ordered plan (search -> read -> compute -> query -> write). Deterministic, offline, reproducible."""
from __future__ import annotations
import re
from typing import Protocol
from pydantic import BaseModel, Field
class LLMResponse(BaseModel):
    thought: str = ""
    tool_name: str | None = None
    tool_args: dict = Field(default_factory=dict)
    final_answer: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
class LLMBackend(Protocol):
    def generate(self, *, task_id: str, goal: str, history: list[dict], available_tools: list[str]) -> LLMResponse:
        ...
def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)
def _plan_search(match: re.Match) -> dict:
    return {"tool": "search", "args": {"query": match.group(1).strip()}}
def _plan_read(match: re.Match) -> dict:
    return {"tool": "read_file", "args": {"path": match.group(1).strip()}}
def _plan_write(match: re.Match) -> dict:
    return {"tool": "write_file", "args": {"path": match.group(1).strip(), "content": match.group(2)}}
def _plan_db(match: re.Match) -> dict:
    return {"tool": "db_query", "args": {"sql": match.group(1).strip()}}
def _plan_compute(match: re.Match) -> dict:
    return {"tool": "calculator", "args": {"expression": match.group(1).strip()}}
def _plan_average(match: re.Match) -> dict:
    numbers = re.findall(r"\d+(?:\.\d+)?", match.group(1))
    expr = "(" + "+".join(numbers) + f")/{len(numbers)}"
    return {"tool": "calculator", "args": {"expression": expr}}
def _plan_sum(match: re.Match) -> dict:
    numbers = re.findall(r"\d+(?:\.\d+)?", match.group(1))
    return {"tool": "calculator", "args": {"expression": "+".join(numbers)}}
def _plan_list(match: re.Match) -> dict:
    return {"tool": "list_files", "args": {}}
_GOAL_PATTERNS: list[tuple[re.Pattern, object]] = [
    (re.compile(r"average of (.+?) and report", re.IGNORECASE), _plan_average),
    (re.compile(r"sum of (.+?) and report", re.IGNORECASE), _plan_sum),
    (re.compile(r"search for '([^']+)'", re.IGNORECASE), _plan_search),
    (re.compile(r"read the file '([^']+)'", re.IGNORECASE), _plan_read),
    (re.compile(r"(?:create|write) (?:a )?file '([^']+)' containing '([^']+)'", re.IGNORECASE), _plan_write),
    (re.compile(r"run the query '([^']+)'", re.IGNORECASE), _plan_db),
    (re.compile(r"compute\s+([0-9][0-9\.\s\+\-\*/\(\)\%]+?)(?=\s+and\s+|\s*,|\.$|$)", re.IGNORECASE), _plan_compute),
    (re.compile(r"\blist (?:the )?files\b", re.IGNORECASE), _plan_list),
]
def build_plan(goal: str, available_tools: list[str]) -> list[dict]:
    found: list[tuple[int, dict]] = []
    for pattern, builder in _GOAL_PATTERNS:
        for match in pattern.finditer(goal):
            step = builder(match)
            if step["tool"] in available_tools:
                found.append((match.start(), step))
    found.sort(key=lambda item: item[0])
    plan: list[dict] = []
    for _, step in found:
        if not plan or plan[-1] != step:
            plan.append(step)
    return plan
def _summarise_step(tool_name: str, observation: str) -> str:
    if tool_name == "calculator":
        m = re.search(r"Result:\s*(.+)", observation)
        return f"The result is {m.group(1).strip()}." if m else observation
    if tool_name == "search":
        m = re.search(r"\d+\.\s+[^\n—]+—\s*(.+)", observation)
        return m.group(1).strip() if m else observation
    if tool_name == "read_file":
        _, _, content = observation.partition("\n")
        return f"The file contains: {content.strip()}"
    if tool_name == "write_file":
        m = re.search(r"to (\S+)\.", observation)
        return f"I created the file {m.group(1)}." if m else "I created the file."
    if tool_name == "db_query":
        _, _, rows = observation.partition("\n")
        return f"The query returned: {rows.strip()}"
    return observation
class MockLLMBackend:
    def __init__(self) -> None:
        self._plans: dict[str, list[dict]] = {}
    def generate(self, *, task_id: str, goal: str, history: list[dict], available_tools: list[str]) -> LLMResponse:
        if task_id not in self._plans:
            self._plans[task_id] = build_plan(goal, available_tools)
        plan = self._plans[task_id]
        step_index = len(history)
        prompt_text = goal + "\n" + "\n".join(f"{h['tool']}({h['args']}) -> {h['observation']}" for h in history)
        input_tokens = _count_tokens(prompt_text)
        if step_index < len(plan):
            step = plan[step_index]
            thought = f"Step {step_index + 1}: I will use {step['tool']} with {step['args']}."
            response_text = thought + str(step)
            return LLMResponse(thought=thought, tool_name=step["tool"], tool_args=dict(step["args"]), input_tokens=input_tokens, output_tokens=_count_tokens(response_text),)
        if not plan:
            answer = "I cannot complete this task with the available tools."
        else:
            parts = [_summarise_step(h["tool"], h["observation"]) for h in history if h.get("observation") and not h["observation"].startswith("Error")]
            answer = " ".join(parts) if parts else "Task complete."
        thought = "I have gathered the information I need; I will now answer."
        return LLMResponse(thought=thought, final_answer=answer, input_tokens=input_tokens, output_tokens=_count_tokens(thought + answer),)
