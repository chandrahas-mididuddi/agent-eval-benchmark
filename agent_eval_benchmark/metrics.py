"""Metrics: per-task scores and aggregate reports. Cost $3/1M input, $15/1M output."""
from __future__ import annotations
from agent_eval_benchmark.agent import AgentTrace
from agent_eval_benchmark.tasks import Task
INPUT_COST_PER_MILLION_TOKENS = 3.0
OUTPUT_COST_PER_MILLION_TOKENS = 15.0
def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * INPUT_COST_PER_MILLION_TOKENS / 1_000_000 + output_tokens * OUTPUT_COST_PER_MILLION_TOKENS / 1_000_000)
def tool_call_accuracy(trace: AgentTrace, task: Task) -> float:
    if task.expected_calls:
        matched = 0
        for expected in task.expected_calls:
            for name, args, ok in trace.tool_calls:
                if name == expected.tool and ok and all(args.get(k) == v for k, v in expected.args.items()):
                    matched += 1
                    break
        return matched / len(task.expected_calls)
    calls = trace.tool_calls
    if not calls:
        return 0.0
    correct = sum(1 for name, _args, ok in calls if name in task.tools and ok)
    return correct / len(calls)
def task_metrics(trace: AgentTrace, task: Task, passed: bool) -> dict:
    cost = estimate_cost(trace.total_input_tokens, trace.total_output_tokens)
    return {"task_id": task.id, "category": task.category, "difficulty": task.difficulty, "success": passed, "completed": trace.completed, "steps": len(trace.steps), "tool_call_accuracy": round(tool_call_accuracy(trace, task), 4), "input_tokens": trace.total_input_tokens, "output_tokens": trace.total_output_tokens, "total_tokens": trace.total_input_tokens + trace.total_output_tokens, "cost_usd": round(cost, 6), "latency_s": round(trace.total_latency_s, 4),}
def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n_tasks": 0}
    successes = sum(1 for r in results if r["success"])
    per_category: dict[str, dict] = {}
    for row in results:
        bucket = per_category.setdefault(row["category"], {"n": 0, "successes": 0})
        bucket["n"] += 1
        bucket["successes"] += int(row["success"])
    for bucket in per_category.values():
        bucket["success_rate"] = round(bucket["successes"] / bucket["n"], 4)
    return {"n_tasks": n, "success_rate": round(successes / n, 4), "successes": successes, "avg_steps": round(sum(r["steps"] for r in results) / n, 2), "avg_tool_call_accuracy": round(sum(r["tool_call_accuracy"] for r in results) / n, 4), "total_tokens": sum(r["total_tokens"] for r in results), "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6), "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 4), "per_category": per_category,}
