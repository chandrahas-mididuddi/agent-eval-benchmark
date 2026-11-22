"""Reference agent: ReAct-style loop over mock tools. Stops on final answer or max_steps (trace incomplete then)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from agent_eval_benchmark.llm import LLMBackend
from agent_eval_benchmark.tasks import Task
from agent_eval_benchmark.tools import ToolEnvironment
@dataclass
class AgentStep:
    index: int
    thought: str
    tool_name: str
    tool_args: dict
    observation: str
    ok: bool
    input_tokens: int
    output_tokens: int
    latency_s: float
@dataclass
class AgentTrace:
    task_id: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    completed: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_s: float = 0.0
    @property
    def tool_calls(self) -> list[tuple[str, dict, bool]]:
        return [(s.tool_name, s.tool_args, s.ok) for s in self.steps]
class ReActAgent:
    def __init__(self, llm: LLMBackend, max_steps: int = 10, allowed_tools: list[str] | None = None) -> None:
        self.llm = llm
        self.max_steps = max_steps
        self.allowed_tools = allowed_tools
    def run(self, task: Task, env: ToolEnvironment) -> AgentTrace:
        available_tools = ([t for t in task.tools if t in self.allowed_tools] if self.allowed_tools is not None else list(task.tools))
        trace = AgentTrace(task_id=task.id)
        history: list[dict] = []
        limit = min(self.max_steps, task.max_steps)
        for index in range(limit):
            started = time.perf_counter()
            response = self.llm.generate(task_id=task.id, goal=task.goal, history=history, available_tools=available_tools,)
            if response.final_answer is not None:
                trace.final_answer = response.final_answer
                trace.completed = True
                trace.total_input_tokens += response.input_tokens
                trace.total_output_tokens += response.output_tokens
                trace.total_latency_s += time.perf_counter() - started
                break
            if response.tool_name is None:
                trace.final_answer = ""
                trace.completed = False
                break
            if response.tool_name not in available_tools:
                observation = f"Error: tool {response.tool_name!r} is not available for this task."
                ok = False
            else:
                result = env.call(response.tool_name, response.tool_args)
                observation, ok = result.output, result.ok
            latency = time.perf_counter() - started
            trace.steps.append(AgentStep(index=index, thought=response.thought, tool_name=response.tool_name, tool_args=dict(response.tool_args), observation=observation, ok=ok, input_tokens=response.input_tokens, output_tokens=response.output_tokens, latency_s=latency,))
            trace.total_input_tokens += response.input_tokens
            trace.total_output_tokens += response.output_tokens
            trace.total_latency_s += latency
            history.append({"tool": response.tool_name, "args": dict(response.tool_args), "observation": observation,})
        return trace
