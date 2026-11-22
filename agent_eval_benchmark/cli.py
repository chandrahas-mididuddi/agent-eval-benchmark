"""CLI: python -m agent_eval_benchmark.cli run --tasks tasks/tasks.yaml [--category math-data] [--seed 42] / compare --config-a ... --config-b ..."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from agent_eval_benchmark.runner import (AgentConfig, compare_evals, run_eval, write_comparison_markdown, write_json_report, write_markdown_report,)
from agent_eval_benchmark.tasks import load_tasks
def _load_config(path: str | None, *, name: str, max_steps: int | None) -> AgentConfig:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return AgentConfig.from_dict(data)
    return AgentConfig(name=name, max_steps=max_steps or 10)
def _filter(tasks, category: str | None, difficulty: str | None):
    if category:
        tasks = [t for t in tasks if t.category == category]
    if difficulty:
        tasks = [t for t in tasks if t.difficulty == difficulty]
    return tasks
def _print_summary(result) -> None:
    agg = result.aggregate
    print(f"\nConfig: {result.config.name}  (seed {result.seed})")
    print(f"Tasks: {agg.get('n_tasks', 0)}")
    print(f"Success rate: {agg.get('success_rate', 0):.1%} ({agg.get('successes', 0)}/{agg.get('n_tasks', 0)})")
    print(f"Avg steps: {agg.get('avg_steps', 0)}")
    print(f"Avg tool-call accuracy: {agg.get('avg_tool_call_accuracy', 0):.2f}")
    print(f"Total tokens: {agg.get('total_tokens', 0)}")
    print(f"Total cost: ${agg.get('total_cost_usd', 0):.4f}")
    print(f"Avg latency: {agg.get('avg_latency_s', 0)}s")
def cmd_run(args: argparse.Namespace) -> int:
    tasks = _filter(load_tasks(args.tasks), args.category, args.difficulty)
    if not tasks:
        print("No tasks matched the given filters.", file=sys.stderr)
        return 1
    config = _load_config(args.config, name=args.name, max_steps=args.max_steps)
    result = run_eval(tasks, config, seed=args.seed)
    out_dir = Path(args.output)
    write_json_report(result, out_dir / "results.json")
    md_path = write_markdown_report(result, out_dir / "report.md")
    _print_summary(result)
    print(f"\nReports written to {out_dir}/ (results.json, report.md)")
    print(md_path.read_text(encoding="utf-8"))
    return 0
def cmd_compare(args: argparse.Namespace) -> int:
    tasks = _filter(load_tasks(args.tasks), args.category, args.difficulty)
    if not tasks:
        print("No tasks matched the given filters.", file=sys.stderr)
        return 1
    config_a = _load_config(args.config_a, name="config-a", max_steps=None)
    config_b = _load_config(args.config_b, name="config-b", max_steps=None)
    result_a = run_eval(tasks, config_a, seed=args.seed)
    result_b = run_eval(tasks, config_b, seed=args.seed)
    comparison = compare_evals(result_a, result_b)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    md_path = write_comparison_markdown(comparison, out_dir / "comparison.md")
    print(f"\n{config_a.name} success rate: {result_a.aggregate.get('success_rate', 0):.1%}")
    print(f"{config_b.name} success rate: {result_b.aggregate.get('success_rate', 0):.1%}")
    print(f"Delta (A - B): {comparison['success_rate_delta']:+.1%}")
    print(f"\nComparison written to {out_dir}/ (comparison.json, comparison.md)")
    print(md_path.read_text(encoding="utf-8"))
    return 0
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-eval-benchmark", description="Evaluate agentic AI workflows.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run the task suite under one agent config.")
    run_p.add_argument("--tasks", default="tasks/tasks.yaml")
    run_p.add_argument("--config", help="JSON file with an agent config")
    run_p.add_argument("--name", default="default", help="Config name (if no --config)")
    run_p.add_argument("--max-steps", type=int, help="Max agent steps (if no --config)")
    run_p.add_argument("--category", help="Only run tasks in this category")
    run_p.add_argument("--difficulty", help="Only run tasks of this difficulty")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--output", default="reports")
    run_p.set_defaults(func=cmd_run)
    cmp_p = sub.add_parser("compare", help="Run two agent configs and compare them side by side.")
    cmp_p.add_argument("--tasks", default="tasks/tasks.yaml")
    cmp_p.add_argument("--config-a", required=True, help="JSON file for config A")
    cmp_p.add_argument("--config-b", required=True, help="JSON file for config B")
    cmp_p.add_argument("--category", help="Only run tasks in this category")
    cmp_p.add_argument("--difficulty", help="Only run tasks of this difficulty")
    cmp_p.add_argument("--seed", type=int, default=42)
    cmp_p.add_argument("--output", default="reports")
    cmp_p.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return args.func(args)
if __name__ == "__main__":
    raise SystemExit(main())
