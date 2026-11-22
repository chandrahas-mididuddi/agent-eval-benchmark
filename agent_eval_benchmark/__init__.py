"""agent-eval-benchmark: a reproducible evaluation benchmark for agentic AI workflows. Measures task success, tool-call accuracy, token cost, and latency across agent runs, using a deterministic mock tool environment so evals run fully offline and reproducibly."""
from agent_eval_benchmark.runner import AgentConfig, EvalResult, compare_evals, run_eval
from agent_eval_benchmark.tasks import Checker, Task, load_tasks
__all__ = ["AgentConfig", "Checker", "EvalResult", "Task", "compare_evals", "load_tasks", "run_eval",]
__version__ = "0.1.0"
