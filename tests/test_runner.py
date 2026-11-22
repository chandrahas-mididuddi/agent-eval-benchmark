"""Tests: metrics math, agent loop, runner reproducibility, comparison."""
from agent_eval_benchmark.metrics import aggregate, estimate_cost, tool_call_accuracy
from agent_eval_benchmark.runner import AgentConfig, compare_evals, run_eval
from agent_eval_benchmark.tasks import load_tasks
def test_estimate_cost():
    assert estimate_cost(1_000_000, 0) == 3.0
    assert estimate_cost(0, 1_000_000) == 15.0
def test_aggregate_empty():
    assert aggregate([]) == {"n_tasks": 0}
def test_run_eval_baseline():
    tasks = load_tasks("tasks/tasks.yaml")
    result = run_eval(tasks, AgentConfig(name="baseline", max_steps=10), seed=42)
    assert result.aggregate["n_tasks"] == 12
    assert result.aggregate["success_rate"] == 1.0
def test_run_eval_low_steps_fails_plan03():
    tasks = load_tasks("tasks/tasks.yaml")
    result = run_eval(tasks, AgentConfig(name="low-steps", max_steps=3), seed=42)
    assert result.aggregate["success_rate"] < 1.0
    failed = [r for r in result.per_task if not r["success"]]
    assert any(r["task_id"] == "plan-03" for r in failed)
def test_run_eval_no_search():
    tasks = load_tasks("tasks/tasks.yaml")
    result = run_eval(tasks, AgentConfig(name="no-search", max_steps=10, allowed_tools=["calculator", "read_file", "write_file", "list_files", "db_query"]), seed=42)
    assert result.aggregate["success_rate"] == 0.75
def test_reproducibility():
    tasks = load_tasks("tasks/tasks.yaml")
    a = run_eval(tasks, AgentConfig(name="baseline", max_steps=10), seed=42)
    b = run_eval(tasks, AgentConfig(name="baseline", max_steps=10), seed=42)
    det_a = [(r["task_id"], r["success"], r["steps"], r["total_tokens"]) for r in a.per_task]
    det_b = [(r["task_id"], r["success"], r["steps"], r["total_tokens"]) for r in b.per_task]
    assert det_a == det_b
def test_compare_evals():
    tasks = load_tasks("tasks/tasks.yaml")
    a = run_eval(tasks, AgentConfig(name="baseline", max_steps=10), seed=42)
    b = run_eval(tasks, AgentConfig(name="low-steps", max_steps=3), seed=42)
    comp = compare_evals(a, b)
    assert comp["success_rate_delta"] > 0
    assert len(comp["rows"]) == 12
