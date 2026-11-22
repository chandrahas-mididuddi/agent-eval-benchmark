"""Tests: every checker type, load_tasks, evaluate_task."""
from agent_eval_benchmark.agent import AgentStep, AgentTrace
from agent_eval_benchmark.tasks import Checker, Task, evaluate_checker, evaluate_task, load_tasks
def _trace(answer="Paris is the capital.", steps=None):
    return AgentTrace(task_id="t1", steps=steps or [], final_answer=answer, completed=True)
def test_load_tasks():
    tasks = load_tasks("tasks/tasks.yaml")
    assert len(tasks) == 12
    assert tasks[0].id == "web-01"
def test_final_answer_contains():
    c = Checker(type="final_answer_contains", value="Paris")
    assert evaluate_checker(c, _trace(), None)[0] is True
    c2 = Checker(type="final_answer_contains", value="London")
    assert evaluate_checker(c2, _trace(), None)[0] is False
def test_final_answer_equals():
    c = Checker(type="final_answer_equals", value="Paris")
    assert evaluate_checker(c, _trace("Paris"), None)[0] is True
def test_file_checkers(tmp_path):
    from agent_eval_benchmark.tools import ToolEnvironment
    env = ToolEnvironment()
    try:
        env.reset({"a.txt": "hello world"})
        assert evaluate_checker(Checker(type="file_exists", path="a.txt"), _trace(), env)[0] is True
        assert evaluate_checker(Checker(type="file_content_contains", path="a.txt", value="world"), _trace(), env)[0] is True
        assert evaluate_checker(Checker(type="file_content_equals", path="a.txt", value="hello world"), _trace(), env)[0] is True
        assert evaluate_checker(Checker(type="file_exists", path="nope.txt"), _trace(), env)[0] is False
    finally:
        env.close()
def test_tool_called_checker():
    step = AgentStep(index=0, thought="t", tool_name="search", tool_args={"query": "capital of France"}, observation="Paris", ok=True, input_tokens=1, output_tokens=1, latency_s=0.0)
    assert evaluate_checker(Checker(type="tool_called", tool="search", args={"query": "capital of France"}), _trace(steps=[step]), None)[0] is True
    assert evaluate_checker(Checker(type="tool_called", tool="calculator", args={}), _trace(steps=[step]), None)[0] is False
def test_all_of_checker():
    c = Checker(type="all_of", checkers=[Checker(type="final_answer_contains", value="Paris"), Checker(type="final_answer_contains", value="capital")])
    assert evaluate_checker(c, _trace(), None)[0] is True
def test_unknown_checker_raises():
    import pytest
    with pytest.raises(ValueError):
        evaluate_checker(Checker(type="nope"), _trace(), None)
def test_evaluate_task_all_pass():
    task = Task(id="t", category="web-research", difficulty="easy", goal="g", tools=["search"], checker=[Checker(type="final_answer_contains", value="Paris")])
    passed, results = evaluate_task(task, _trace(), None)
    assert passed is True and len(results) == 1
