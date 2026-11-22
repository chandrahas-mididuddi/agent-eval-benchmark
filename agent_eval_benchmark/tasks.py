"""Task suite format, loading, and deterministic checkers. No LLM-as-judge in scoring — checkers are assertions over final answer, tool calls, workspace state."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field
class Checker(BaseModel):
    type: str
    value: str | None = None
    path: str | None = None
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    checkers: list["Checker"] = Field(default_factory=list)
Checker.model_rebuild()
class ExpectedCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
class Task(BaseModel):
    id: str
    category: str
    difficulty: str
    goal: str
    tools: list[str]
    max_steps: int = 10
    initial_files: dict[str, str] = Field(default_factory=dict)
    checker: list[Checker] = Field(default_factory=list)
    expected_calls: list[ExpectedCall] = Field(default_factory=list)
def load_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"task file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"unsupported task file format: {path.suffix}")
    if not isinstance(raw, list):
        raise ValueError("task file must contain a list of tasks")
    tasks = [Task(**item) for item in raw]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate task ids in {path}")
    return tasks
def evaluate_checker(checker: Checker, trace: Any, env: Any) -> tuple[bool, str]:
    final_answer = (getattr(trace, "final_answer", None) or "")
    if checker.type == "final_answer_contains":
        passed = checker.value in final_answer
        return passed, f"final answer {'contains' if passed else 'does not contain'} {checker.value!r}"
    if checker.type == "final_answer_equals":
        passed = final_answer.strip() == (checker.value or "")
        return passed, f"final answer {'equals' if passed else 'does not equal'} {checker.value!r}"
    if checker.type == "file_exists":
        passed = env.file_exists(checker.path or "")
        return passed, f"file {checker.path!r} {'exists' if passed else 'is missing'}"
    if checker.type == "file_content_contains":
        try:
            content = env.read_file_text(checker.path or "")
        except (FileNotFoundError, OSError):
            return False, f"file {checker.path!r} is missing"
        passed = (checker.value or "") in content
        return passed, f"file {checker.path!r} {'contains' if passed else 'does not contain'} {checker.value!r}"
    if checker.type == "file_content_equals":
        try:
            content = env.read_file_text(checker.path or "")
        except (FileNotFoundError, OSError):
            return False, f"file {checker.path!r} is missing"
        passed = content == (checker.value or "")
        return passed, f"file {checker.path!r} {'equals' if passed else 'does not equal'} {checker.value!r}"
    if checker.type == "tool_called":
        for step in getattr(trace, "steps", []):
            if step.tool_name != checker.tool:
                continue
            if all(step.tool_args.get(k) == v for k, v in checker.args.items()):
                return True, f"tool {checker.tool!r} called with matching args"
        return False, f"tool {checker.tool!r} never called with args {checker.args}"
    if checker.type == "all_of":
        details: list[str] = []
        for sub in checker.checkers:
            passed, detail = evaluate_checker(sub, trace, env)
            details.append(detail)
            if not passed:
                return False, "all_of failed: " + "; ".join(details)
        return True, "all_of passed: " + "; ".join(details)
    raise ValueError(f"unknown checker type: {checker.type!r}")
def evaluate_task(task: Task, trace: Any, env: Any) -> tuple[bool, list[dict[str, Any]]]:
    results = []
    for checker in task.checker:
        passed, detail = evaluate_checker(checker, trace, env)
        results.append({"type": checker.type, "passed": passed, "detail": detail})
    return all(r["passed"] for r in results), results
