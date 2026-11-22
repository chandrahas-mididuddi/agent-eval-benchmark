"""Eval runner: executes task suite under one agent config, reproducible with same seed/config."""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from agent_eval_benchmark.agent import ReActAgent
from agent_eval_benchmark.llm import MockLLMBackend
from agent_eval_benchmark.metrics import aggregate, task_metrics
from agent_eval_benchmark.tasks import Task, evaluate_task
from agent_eval_benchmark.tools import ToolEnvironment
@dataclass
class AgentConfig:
    name: str = "default"
    max_steps: int = 10
    allowed_tools: list[str] | None = None
    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        return cls(name=data.get("name", "default"), max_steps=int(data.get("max_steps", 10)), allowed_tools=data.get("allowed_tools"),)
    def to_dict(self) -> dict:
        return {"name": self.name, "max_steps": self.max_steps, "allowed_tools": self.allowed_tools,}
@dataclass
class EvalResult:
    config: AgentConfig
    seed: int
    per_task: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    def to_dict(self) -> dict:
        return {"config": self.config.to_dict(), "seed": self.seed, "aggregate": self.aggregate, "per_task": self.per_task,}
def run_eval(tasks: list[Task], config: AgentConfig, *, seed: int = 42, llm_factory=None) -> EvalResult:
    random.seed(seed)
    llm = llm_factory() if llm_factory is not None else MockLLMBackend()
    agent = ReActAgent(llm, max_steps=config.max_steps, allowed_tools=config.allowed_tools)
    per_task: list[dict] = []
    for task in tasks:
        env = ToolEnvironment(seed=seed)
        try:
            env.reset(task.initial_files)
            trace = agent.run(task, env)
            passed, checker_results = evaluate_task(task, trace, env)
            row = task_metrics(trace, task, passed)
            row["final_answer"] = trace.final_answer
            row["checkers"] = checker_results
            per_task.append(row)
        finally:
            env.close()
    return EvalResult(config=config, seed=seed, per_task=per_task, aggregate=aggregate(per_task))
def compare_evals(a: EvalResult, b: EvalResult) -> dict:
    by_id_a = {r["task_id"]: r for r in a.per_task}
    by_id_b = {r["task_id"]: r for r in b.per_task}
    rows = []
    for task_id in sorted(set(by_id_a) | set(by_id_b)):
        ra, rb = by_id_a.get(task_id), by_id_b.get(task_id)
        rows.append({"task_id": task_id, "category": (ra or rb)["category"], f"success_{a.config.name}": (ra or {}).get("success"), f"success_{b.config.name}": (rb or {}).get("success"), f"steps_{a.config.name}": (ra or {}).get("steps"), f"steps_{b.config.name}": (rb or {}).get("steps"), f"cost_usd_{a.config.name}": (ra or {}).get("cost_usd"), f"cost_usd_{b.config.name}": (rb or {}).get("cost_usd"),})
    return {"config_a": a.config.to_dict(), "config_b": b.config.to_dict(), "seed": a.seed, "aggregate_a": a.aggregate, "aggregate_b": b.aggregate, "success_rate_delta": round(a.aggregate.get("success_rate", 0) - b.aggregate.get("success_rate", 0), 4), "rows": rows,}
def write_json_report(result: EvalResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
def _task_table(per_task: list[dict]) -> str:
    lines = ["| Task | Category | Difficulty | Success | Steps | Tool Acc. | Tokens | Cost ($) | Latency (s) |", "|------|----------|------------|---------|-------|-----------|--------|----------|-------------|",]
    for r in per_task:
        lines.append(f"| {r['task_id']} | {r['category']} | {r['difficulty']} | {'✅' if r['success'] else '❌'} | {r['steps']} | {r['tool_call_accuracy']:.2f} | {r['total_tokens']} | {r['cost_usd']:.4f} | {r['latency_s']:.3f} |")
    return "\n".join(lines)
def write_markdown_report(result: EvalResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = result.aggregate
    lines = [f"# Eval Report — {result.config.name}", "", f"- **Seed:** {result.seed}", f"- **Tasks:** {agg.get('n_tasks', 0)}", f"- **Success rate:** {agg.get('success_rate', 0):.1%} ({agg.get('successes', 0)}/{agg.get('n_tasks', 0)})", f"- **Avg steps:** {agg.get('avg_steps', 0)}", f"- **Avg tool-call accuracy:** {agg.get('avg_tool_call_accuracy', 0):.2f}", f"- **Total tokens:** {agg.get('total_tokens', 0)}", f"- **Total cost:** ${agg.get('total_cost_usd', 0):.4f}", f"- **Avg latency:** {agg.get('avg_latency_s', 0)}s", "", "## Per-category success rate", "", "| Category | Tasks | Success rate |", "|----------|-------|--------------|",]
    for category, bucket in sorted(agg.get("per_category", {}).items()):
        lines.append(f"| {category} | {bucket['n']} | {bucket['success_rate']:.1%} |")
    lines += ["", "## Per-task results", "", _task_table(result.per_task), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
def write_comparison_markdown(comparison: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name_a = comparison["config_a"]["name"]
    name_b = comparison["config_b"]["name"]
    agg_a, agg_b = comparison["aggregate_a"], comparison["aggregate_b"]
    lines = [f"# Comparison — {name_a} vs {name_b}", "", f"- **{name_a}:** max_steps={comparison['config_a']['max_steps']}, allowed_tools={comparison['config_a']['allowed_tools']}", f"- **{name_b}:** max_steps={comparison['config_b']['max_steps']}, allowed_tools={comparison['config_b']['allowed_tools']}", "", "| Metric | " + name_a + " | " + name_b + " |", "|--------|" + "-" * (len(name_a) + 2) + "|" + "-" * (len(name_b) + 2) + "|", f"| Success rate | {agg_a.get('success_rate', 0):.1%} | {agg_b.get('success_rate', 0):.1%} |", f"| Avg steps | {agg_a.get('avg_steps', 0)} | {agg_b.get('avg_steps', 0)} |", f"| Avg tool-call accuracy | {agg_a.get('avg_tool_call_accuracy', 0):.2f} | {agg_b.get('avg_tool_call_accuracy', 0):.2f} |", f"| Total tokens | {agg_a.get('total_tokens', 0)} | {agg_b.get('total_tokens', 0)} |", f"| Total cost ($) | {agg_a.get('total_cost_usd', 0):.4f} | {agg_b.get('total_cost_usd', 0):.4f} |", f"| Avg latency (s) | {agg_a.get('avg_latency_s', 0)} | {agg_b.get('avg_latency_s', 0)} |", "", "## Per-task comparison", "", "| Task | Category | Success " + name_a + " | Success " + name_b + " | Steps " + name_a + " | Steps " + name_b + " |",]
    for row in comparison["rows"]:
        sa = "✅" if row[f"success_{name_a}"] else "❌"
        sb = "✅" if row[f"success_{name_b}"] else "❌"
        lines.append(f"| {row['task_id']} | {row['category']} | {sa} | {sb} | {row[f'steps_{name_a}']} | {row[f'steps_{name_b}']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
