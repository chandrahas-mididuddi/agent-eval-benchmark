"""Tests: calculator safety, file sandboxing, search, SQL subset."""
from agent_eval_benchmark.tools import ToolEnvironment, evaluate_expression, format_number
import pytest
def test_calculator_basic():
    assert evaluate_expression("128 * 46 + 320") == 6208
    assert evaluate_expression("2 + 3 * 4") == 14
def test_calculator_rejects_injection():
    with pytest.raises(Exception):
        evaluate_expression("__import__('os').system('ls')")
    with pytest.raises(Exception):
        evaluate_expression("1/0")
def test_format_number():
    assert format_number(250.0) == "250"
    assert format_number(2.5) == "2.5"
def test_file_sandbox_blocks_escape(tmp_path):
    env = ToolEnvironment()
    try:
        assert env.call("read_file", {"path": "../secret.txt"}).ok is False
        assert env.call("read_file", {"path": "/etc/passwd"}).ok is False
        env.call("write_file", {"path": "a.txt", "content": "hi"})
        assert env.read_file_text("a.txt") == "hi"
        assert env.file_exists("a.txt") is True
    finally:
        env.close()
def test_search_deterministic():
    env = ToolEnvironment()
    try:
        r = env.call("search", {"query": "capital of France"})
        assert r.ok and "Paris" in r.output
        r2 = env.call("search", {"query": "capital of France"})
        assert r.output == r2.output
    finally:
        env.close()
def test_db_query_sum():
    env = ToolEnvironment()
    try:
        r = env.call("db_query", {"sql": "SELECT SUM(amount) FROM orders WHERE status = shipped"})
        assert r.ok and "950" in r.output
        r2 = env.call("db_query", {"sql": "SELECT COUNT(*) FROM orders"})
        assert r2.ok and "4" in r2.output
    finally:
        env.close()
def test_unknown_tool():
    env = ToolEnvironment()
    try:
        assert env.call("nope", {}).ok is False
    finally:
        env.close()
