# agent-eval-benchmark

A reproducible evaluation benchmark for agentic AI workflows. It measures **task
success rate, tool-call accuracy, token cost, and latency** across agent runs —
using a deterministic mock tool environment, so evals run fully offline and
reproduce exactly.

This is a portfolio project. It is not affiliated with any company, and nothing
here is presented as production software. It exists to show how an agent
evaluation harness is designed, built, and tested.

## What it does

- **Task suite** (`tasks/tasks.yaml`): 12 tasks across four categories —
  web research (mock), file manipulation, math/data analysis, and multi-step
  planning — each with a goal, allowed tools, an optional starting workspace,
  and deterministic checkers.
- **Mock tool environment** (`agent_eval_benchmark/tools.py`): calculator
  (safe AST evaluation, no `eval()`), sandboxed file read/write, a mock search
  API, and a mock database with a small read-only SQL subset. Deterministic,
  no network needed.
- **Reference agent** (`agent_eval_benchmark/agent.py`): a ReAct-style loop
  that runs against the tools with a pluggable LLM backend. Ships with a
  deterministic rule-based mock backend (`llm.py`) so evals are reproducible
  offline; swap in a real model by implementing the `LLMBackend` protocol.
- **Metrics** (`metrics.py`): success rate, tool-call accuracy (right tool +
  right args), average steps, tokens, cost estimate ($3/1M input, $15/1M
  output tokens by default), and latency — per task and in aggregate.
- **Eval runner + CLI** (`runner.py`, `cli.py`): run the suite, filter by
  category/difficulty, set a seed, and write JSON + Markdown reports. A
  **comparison mode** runs two agent configs side by side.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m agent_eval_benchmark.cli run --tasks tasks/tasks.yaml --seed 42
python -m agent_eval_benchmark.cli run --tasks tasks/tasks.yaml --category math-data
python -m agent_eval_benchmark.cli compare --tasks tasks/tasks.yaml --config-a configs/baseline.json --config-b configs/low-steps.json
```

Reports land in `reports/` (`results.json` + `report.md`, or
`comparison.json` + `comparison.md`).

## Sample report

Baseline config (10 max steps, all tools), seed 42 — actual output:

```
Config: baseline  (seed 42)
Tasks: 12
Success rate: 100.0% (12/12)
Avg steps: 1.75
Avg tool-call accuracy: 1.00
Total tokens: 3396
Total cost: $0.0239
Avg latency: 0.0003s
```

Comparison against a 3-step limit shows exactly where the budget bites:

| Metric | baseline | low-steps |
|--------|----------|-----------|
| Success rate | 100.0% | 91.7% |
| Avg steps | 1.75 | 1.67 |
| Total tokens | 3396 | 2666 |
| Total cost ($) | 0.0239 | 0.0196 |

The only failure is `plan-03` (a 4-step research → compute → write task) —
it needs 4 steps and the low-steps config allows 3. Removing the search tool
entirely (`configs/no-search.json`) drops success to 75%, since all three
web-research tasks fail. That is the point of the benchmark: a config change
should move a number, and here it does, in a way you can explain.

## Benchmark design

**Why deterministic checkers, not LLM-as-judge.** Every checker is an
assertion over the final answer, the tool calls, or the workspace state after
the run (e.g. "file `budget.txt` contains `Monthly total: 1830`"). Scoring
never depends on a model's opinion, so the same run always scores the same —
verified by a reproducibility test that runs the suite twice with the same
seed and requires identical deterministic results.

**Why a mock LLM backend.** Real models are slow, expensive, and
non-deterministic, which makes them a bad default for a benchmark harness.
The mock backend is a rule-based policy that parses the goal into a plan of
tool calls. It is deliberately a *baseline*: it can only do what the goal
spells out, and it fails when the step budget is too small or a needed tool
is removed. The harness measures the agent, not the mock.

**What the numbers mean.**

| Metric | Definition |
|--------|-----------|
| Success rate | Fraction of tasks where every checker passed |
| Tool-call accuracy | Fraction of tool calls that used an allowed tool and succeeded; if a task lists `expected_calls`, the fraction of those expected calls actually made (right tool + matching args) |
| Avg steps | Mean tool calls per task |
| Tokens / cost | Input + output tokens (mock backend estimates ~4 chars/token); cost at $3/1M in, $15/1M out |
| Latency | Wall-clock seconds per task (excluded from reproducibility checks — it is inherently noisy) |

## Adding a new task

Append a record to `tasks/tasks.yaml`. Checker types: `final_answer_contains`, `final_answer_equals`,
`file_exists`, `file_content_contains`, `file_content_equals`,
`tool_called` (with an `args` subset to match), and `all_of` (a list of
nested checkers — all must pass). Optional fields: `initial_files`, `max_steps`, `expected_calls`.

## Plugging in a real LLM backend

Implement the `LLMBackend` protocol — one method `generate(task_id, goal, history, available_tools)` returning an `LLMResponse` with either a tool call or a final answer. Then pass it to `run_eval` with `llm_factory=MyBackend`. Report real token usage from your provider; the mock backend's `len(text) // 4` estimate is only a placeholder.

## Agent configs

| File | What it tests |
|------|--------------|
| `baseline.json` | 10 steps, all tools — the reference point |
| `low-steps.json` | 3 steps — shows which tasks need longer horizons |
| `no-search.json` | No `search` tool — shows which tasks depend on retrieval |

`allowed_tools: null` means "whatever each task allows"; a list restricts
the agent to the intersection of that list and the task's tools.

## Tests

```bash
pytest -q        # 31 tests
```

Coverage: calculator safety, file sandboxing, deterministic search, SQL subset, every checker type, metrics math, the agent loop end to end, runner reproducibility, and the comparison mode. CI runs the suite on Python 3.11 and 3.12 plus a CLI smoke test.

## Limitations

- **Mock backend, mock world.** The search corpus has five facts, the database has two tables, and the mock "LLM" parses goals with regexes.
- **Token counts are estimates** unless a real backend reports usage.
- **Latency is wall-clock time** on the machine running the eval.
- **Checkers are exact.** `final_answer_contains` is substring matching.
- **No parallel execution.** Tasks run sequentially.
- **No retry / error recovery scoring.**

## License

MIT — see [LICENSE](LICENSE).
