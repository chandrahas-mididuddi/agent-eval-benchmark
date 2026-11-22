"""Mock tool environment for agent evaluation. All tools are deterministic and run offline — no network. File operations are sandboxed to a fresh temporary directory per task run, search is an in-memory corpus, database is an in-memory fixture with a small read-only SQL subset."""
from __future__ import annotations
import ast
import copy
import math
import operator
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
class ToolError(Exception):
    """Raised when a tool call is invalid (bad args, missing file, ...)."""
@dataclass
class ToolResult:
    ok: bool
    output: str
    data: object = None
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.FloorDiv: operator.floordiv,}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}
def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ToolError(f"Unsupported expression element: {ast.dump(node)[:80]}")
def evaluate_expression(expression: str) -> float:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("calculator requires a non-empty expression string")
    if re.search(r"[A-Za-z_]{2,}", expression) and not re.search(r"\b(abs|round|min|max|sqrt)\b", expression):
        raise ToolError(f"calculator does not accept variables or words: {expression!r}")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid arithmetic expression: {expression!r}") from exc
    try:
        return _eval_node(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
def format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
SEARCH_CORPUS: dict[str, list[dict[str, str]]] = {
    "capital of france": [{"title": "France — Capital", "snippet": "The capital of France is Paris."}],
    "largest planet": [{"title": "Largest planet in the Solar System", "snippet": "Jupiter is the largest planet in the Solar System."}],
    "population of lagos": [{"title": "Lagos population", "snippet": "Lagos has an estimated population of 15.4 million people."}],
    "founder of microsoft": [{"title": "Microsoft founders", "snippet": "Microsoft was founded by Bill Gates and Paul Allen in 1975."}],
    "speed of light": [{"title": "Speed of light", "snippet": "The speed of light in a vacuum is 299,792,458 metres per second."}],
}
DEFAULT_DB: dict[str, list[dict[str, object]]] = {
    "orders": [{"id": 1, "customer": "Acme", "amount": 250, "status": "shipped"}, {"id": 2, "customer": "Globex", "amount": 400, "status": "shipped"}, {"id": 3, "customer": "Acme", "amount": 150, "status": "pending"}, {"id": 4, "customer": "Initech", "amount": 300, "status": "shipped"},],
    "products": [{"id": 1, "name": "Widget", "price": 25, "stock": 120}, {"id": 2, "name": "Gadget", "price": 40, "stock": 60},],
}
def _parse_value(raw: str) -> object:
    raw = raw.strip().strip("'\"")
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw
def _row_matches(row: dict, clause: str) -> bool:
    m = re.match(r"\s*(\w+)\s*(=|!=|>=|<=|>|<)\s*(.+?)\s*$", clause)
    if not m:
        raise ToolError(f"unsupported WHERE clause: {clause!r}")
    column, op, raw_value = m.groups()
    if column not in row:
        raise ToolError(f"unknown column {column!r}")
    value = _parse_value(raw_value)
    actual = row[column]
    ops = {"=": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le,}
    try:
        return bool(ops[op](actual, value))
    except TypeError as exc:
        raise ToolError(f"cannot compare {actual!r} with {value!r}") from exc
TOOL_SPECS: dict[str, dict] = {
    "calculator": {"description": "Evaluate an arithmetic expression, e.g. '128 * 46 + 320'.", "args": {"expression": "string, arithmetic expression using + - * / % ** ( )"}},
    "read_file": {"description": "Read a text file from the workspace.", "args": {"path": "string, relative file path, e.g. 'notes.txt'"}},
    "write_file": {"description": "Create or overwrite a text file in the workspace.", "args": {"path": "string, relative file path", "content": "string"}},
    "list_files": {"description": "List the files currently in the workspace.", "args": {}},
    "search": {"description": "Search the web (mock index) for a query string.", "args": {"query": "string, search query"}},
    "db_query": {"description": "Run a read-only query against the mock database. Supported: SELECT <cols|COUNT(*)|SUM(col)|AVG(col)> FROM <table> [WHERE <col> <op> <value>].", "args": {"sql": "string, SQL query"}},
}
@dataclass
class ToolEnvironment:
    seed: int = 0
    workdir: str = field(init=False)
    db: dict = field(init=False, default_factory=dict)
    call_log: list[dict] = field(init=False, default_factory=list)
    def __post_init__(self) -> None:
        self.workdir = tempfile.mkdtemp(prefix="agent_eval_")
        self.db = copy.deepcopy(DEFAULT_DB)
    def reset(self, initial_files: dict[str, str] | None = None) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)
        os.makedirs(self.workdir, exist_ok=True)
        self.db = copy.deepcopy(DEFAULT_DB)
        self.call_log = []
        for path, content in (initial_files or {}).items():
            self._write(path, content)
    def close(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)
    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            raise ToolError(f"absolute paths are not allowed: {path!r}")
        full = os.path.normpath(os.path.join(self.workdir, path))
        if not full.startswith(self.workdir):
            raise ToolError(f"path escapes the workspace: {path!r}")
        return full
    def _write(self, path: str, content: str) -> None:
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full) or self.workdir, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
    def file_exists(self, path: str) -> bool:
        try:
            return os.path.isfile(self._resolve(path))
        except ToolError:
            return False
    def read_file_text(self, path: str) -> str:
        with open(self._resolve(path), encoding="utf-8") as fh:
            return fh.read()
    def call(self, name: str, args: dict) -> ToolResult:
        if name not in TOOL_SPECS:
            result = ToolResult(False, f"Error: unknown tool {name!r}.")
            self.call_log.append({"tool": name, "args": args, "ok": False})
            return result
        try:
            result = self._dispatch(name, args or {})
        except ToolError as exc:
            result = ToolResult(False, f"Error: {exc}")
        except Exception as exc:
            result = ToolResult(False, f"Error: unexpected failure in {name}: {exc}")
        self.call_log.append({"tool": name, "args": args, "ok": result.ok})
        return result
    def _dispatch(self, name: str, args: dict) -> ToolResult:
        if name == "calculator":
            expr = args.get("expression")
            if not isinstance(expr, str):
                raise ToolError("calculator requires an 'expression' string")
            value = evaluate_expression(expr)
            return ToolResult(True, f"Result: {format_number(value)}", data=value)
        if name == "read_file":
            path = args.get("path")
            if not isinstance(path, str):
                raise ToolError("read_file requires a 'path' string")
            try:
                content = self.read_file_text(path)
            except FileNotFoundError as exc:
                raise ToolError(f"file not found: {path!r}") from exc
            return ToolResult(True, f"Contents of {path}:\n{content}", data=content)
        if name == "write_file":
            path, content = args.get("path"), args.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                raise ToolError("write_file requires 'path' and 'content' strings")
            self._write(path, content)
            return ToolResult(True, f"Wrote {len(content)} characters to {path}.", data=path)
        if name == "list_files":
            files = sorted(os.path.relpath(os.path.join(root, f), self.workdir) for root, _, names in os.walk(self.workdir) for f in names)
            return ToolResult(True, "Files in workspace: " + (", ".join(files) if files else "(none)"), data=files,)
        if name == "search":
            query = args.get("query")
            if not isinstance(query, str):
                raise ToolError("search requires a 'query' string")
            return self._search(query)
        if name == "db_query":
            sql = args.get("sql")
            if not isinstance(sql, str):
                raise ToolError("db_query requires a 'sql' string")
            return self._db_query(sql)
        raise ToolError(f"tool {name!r} is not implemented")
    def _search(self, query: str) -> ToolResult:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[int, str, list[dict[str, str]]]] = []
        for key, results in SEARCH_CORPUS.items():
            key_tokens = set(re.findall(r"[a-z0-9]+", key))
            overlap = len(tokens & key_tokens)
            if overlap:
                scored.append((overlap, key, results))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if not scored:
            return ToolResult(True, f"No results found for '{query}'.", data=[])
        lines = [f"Search results for '{query}':"]
        for i, (_, _, results) in enumerate(scored[:3], start=1):
            for result in results:
                lines.append(f"{i}. {result['title']} — {result['snippet']}")
        return ToolResult(True, "\n".join(lines), data=scored[:3])
    def _db_query(self, sql: str) -> ToolResult:
        m = re.match(r"\s*SELECT\s+(.+?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?\s*$", sql, re.IGNORECASE | re.DOTALL,)
        if not m:
            raise ToolError("unsupported SQL; use SELECT <cols|COUNT(*)|SUM(col)|AVG(col)> FROM <table> [WHERE <col> <op> <value>]")
        select, table, where = m.group(1).strip(), m.group(2), m.group(3)
        if table not in self.db:
            raise ToolError(f"unknown table {table!r}")
        rows = [dict(r) for r in self.db[table]]
        if where:
            rows = [r for r in rows if _row_matches(r, where)]
        agg = re.match(r"(COUNT|SUM|AVG)\((\*|\w+)\)", select, re.IGNORECASE)
        if agg:
            func, column = agg.group(1).upper(), agg.group(2)
            if func == "COUNT":
                value: object = len(rows)
            else:
                if column not in (rows[0] if rows else {}):
                    raise ToolError(f"unknown column {column!r}")
                values = [r[column] for r in rows]
                value = sum(values) if func == "SUM" else (sum(values) / len(values) if values else 0)
            label = f"{func}({column})"
            return ToolResult(True, f"Query returned 1 row(s):\n{label} = {format_number(float(value)) if isinstance(value, float) else value}", data={label: value},)
        if select == "*":
            columns = list(self.db[table][0].keys()) if self.db[table] else []
        else:
            columns = [c.strip() for c in select.split(",")]
            for column in columns:
                if column not in self.db[table][0]:
                    raise ToolError(f"unknown column {column!r}")
        lines = [f"Query returned {len(rows)} row(s):"]
        for row in rows:
            lines.append(", ".join(f"{c} = {row[c]}" for c in columns))
        return ToolResult(True, "\n".join(lines), data=rows)
