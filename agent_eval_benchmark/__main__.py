"""Allow ``python -m agent_eval_benchmark`` to run the CLI."""
from agent_eval_benchmark.cli import main
if __name__ == "__main__":
    raise SystemExit(main())
