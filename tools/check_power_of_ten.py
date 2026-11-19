"""Static checks for NASA/JPL's "Power of Ten" rules, adapted to Python.

The original rules (Gerard Holzmann, NASA/JPL, 2006) were written for
safety-critical C. Most translate to Python as habits that make code easy to
review and hard to break. Ruff and mypy already enforce several of them (see
docs/CODING_STANDARD.md); this script checks the ones no off-the-shelf linter
covers:

  P10-1  no recursion, direct (f calls f) or indirect (f calls g, g calls f) within
         one module (simple, analyzable control flow)
  P10-2  no unbounded loops such as `while True:`
  P10-4  functions no longer than 60 lines (one printed page)
  P10-6  no `global` / `nonlocal` and no mutable module-level variables
  P10-8  no exec / eval / __import__ (no code that writes code at runtime)

Usage:
    python tools/check_power_of_ten.py src tools app
Exit code 0 = clean, 1 = violations found (printed as path:line: rule message).
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MAX_FUNCTION_LINES = 60
MAX_FILES = 10_000  # P10-2 applies to this tool too: every loop has a fixed upper bound
FORBIDDEN_CALLS = frozenset({"exec", "eval", "__import__"})
MUTABLE_FACTORIES = frozenset({"list", "dict", "set", "defaultdict", "OrderedDict", "deque"})


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


def _function_length(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Lines from the first statement after the docstring to the end of the function."""
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:] or body  # skip the docstring
    end = node.end_lineno or node.lineno
    return end - body[0].lineno + 1


def _is_constant_true(test: ast.expr) -> bool:
    return isinstance(test, ast.Constant) and bool(test.value) is True


def _is_mutable_value(value: ast.expr) -> bool:
    if isinstance(value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
        return True
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in MUTABLE_FACTORIES


def _calls_itself(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    """Line of a direct recursive call (f calling f, or self.f calling self.f), if any."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == node.name:
            return child.lineno
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.attr == node.name
            and func.value.id in {"self", "cls"}
        ):
            return child.lineno
    return None


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """The module's top-level functions and the methods of its top-level classes ("Class.method")."""
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}.{item.name}"] = item
    return found


def _callees(name: str, node: ast.FunctionDef | ast.AsyncFunctionDef, known: set[str]) -> set[str]:
    """The module's own functions that this function calls directly (itself excluded)."""
    owner = name.rpartition(".")[0]
    called: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in {"self", "cls"}:
            called.add(f"{owner}.{func.attr}")
    return (called & known) - {name}


def _path_back(graph: dict[str, set[str]], start: str) -> list[str] | None:
    """A call path from start back to start through other functions, if one exists.

    Breadth-first search without recursion; every round adds at least one new
    function or stops, so it runs at most len(graph) rounds (rule P10-2).
    """
    parent: dict[str, str] = {callee: start for callee in graph[start]}
    frontier = set(parent)
    for _ in range(len(graph)):
        if start in parent:
            break
        nxt = {c for f in frontier for c in graph[f] if c not in parent}
        for f in sorted(frontier):
            for c in sorted(graph[f]):
                parent.setdefault(c, f)
        frontier = nxt
        if not frontier:
            break
    if start not in parent:
        return None
    path = [start]
    for _ in range(len(graph)):
        path.append(parent[path[-1]])
        if path[-1] == start:
            break
    return list(reversed(path))


def _check_call_cycles(tree: ast.Module, path: str) -> Iterator[Violation]:
    """Indirect recursion: functions of one module that call each other in a cycle."""
    functions = _module_functions(tree)
    graph = {name: _callees(name, node, set(functions)) for name, node in functions.items()}
    for name, node in functions.items():
        cycle = _path_back(graph, name)
        if cycle is not None:
            yield Violation(path, node.lineno, "P10-1", f"function '{name}' calls itself through {' -> '.join(cycle)}")


def _check_functions(tree: ast.Module, path: str) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        length = _function_length(node)
        if length > MAX_FUNCTION_LINES:
            yield Violation(
                path, node.lineno, "P10-4", f"function '{node.name}' is {length} lines (max {MAX_FUNCTION_LINES})"
            )
        line = _calls_itself(node)
        if line is not None:
            yield Violation(path, line, "P10-1", f"function '{node.name}' calls itself (recursion)")


def _check_statements(tree: ast.Module, path: str) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if isinstance(node, ast.While) and _is_constant_true(node.test):
            yield Violation(path, node.lineno, "P10-2", "unbounded loop: give the loop a fixed upper bound")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            keyword = "global" if isinstance(node, ast.Global) else "nonlocal"
            yield Violation(path, node.lineno, "P10-6", f"'{keyword}' statement: pass state explicitly instead")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            yield Violation(path, node.lineno, "P10-8", f"'{node.func.id}()' runs code built at runtime")


def _check_module_state(tree: ast.Module, path: str) -> Iterator[Violation]:
    """Module-level mutable variables are shared global state; UPPER_CASE constants are allowed."""
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and not target.id.isupper() and _is_mutable_value(value):
                yield Violation(
                    path, node.lineno, "P10-6", f"module-level mutable variable '{target.id}' (make it a constant)"
                )


def check_source(source: str, path: str = "<string>") -> list[Violation]:
    """Run every rule on one file's source code."""
    tree = ast.parse(source, filename=path)
    found = [
        *_check_functions(tree, path),
        *_check_call_cycles(tree, path),
        *_check_statements(tree, path),
        *_check_module_state(tree, path),
    ]
    return sorted(found, key=lambda v: (v.path, v.line, v.rule))


def iter_python_files(roots: list[str]) -> Iterator[Path]:
    count = 0
    for root in roots:
        base = Path(root)
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for file in files:
            count += 1
            if count > MAX_FILES:
                raise RuntimeError(f"more than {MAX_FILES} files: refusing to continue")
            yield file


def main(argv: list[str] | None = None) -> int:
    roots = (argv if argv is not None else sys.argv[1:]) or ["src"]
    violations: list[Violation] = []
    for file in iter_python_files(roots):
        violations.extend(check_source(file.read_text(encoding="utf-8"), str(file)))
    for violation in violations:
        print(violation)
    print(f"Power of Ten check: {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
