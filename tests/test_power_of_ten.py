"""Tests for the Power of Ten checker (tools/check_power_of_ten.py)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from tools.check_power_of_ten import MAX_FUNCTION_LINES, check_source, main


def rules(source: str) -> list[str]:
    return [v.rule for v in check_source(textwrap.dedent(source))]


def test_clean_code_passes():
    assert (
        rules(
            """
        LIMITS = {"a": 1}

        def add(a, b):
            for _ in range(3):
                a += b
            return a
        """
        )
        == []
    )


def test_recursion_is_flagged():
    assert rules("def f(n):\n    return f(n - 1)\n") == ["P10-1"]
    assert rules("class A:\n    def g(self):\n        return self.g()\n") == ["P10-1"]


def test_indirect_recursion_is_flagged_with_the_cycle():
    found = check_source("def ping(n):\n    return pong(n)\n\ndef pong(n):\n    return ping(n - 1)\n")
    assert [(v.line, v.rule) for v in found] == [(1, "P10-1"), (4, "P10-1")]
    assert "ping -> pong -> ping" in found[0].message
    methods = "class A:\n    def f(self):\n        return self.g()\n    def g(self):\n        return self.f()\n"
    assert rules(methods) == ["P10-1", "P10-1"]
    # Calls that never come back are fine, however deep they go.
    assert rules("def a():\n    b()\n    c()\n\ndef b():\n    c()\n\ndef c():\n    return 1\n") == []


def test_unbounded_loop_is_flagged_but_bounded_while_is_allowed():
    assert rules("while True:\n    pass\n") == ["P10-2"]
    assert rules("i = 0\nwhile i < 10:\n    i += 1\n") == []


def test_long_function_is_flagged():
    body = "\n".join(f"    x{i} = {i}" for i in range(MAX_FUNCTION_LINES + 1))
    assert rules(f'def long():\n    """Doc."""\n{body}\n') == ["P10-4"]
    short = "\n".join(f"    x{i} = {i}" for i in range(MAX_FUNCTION_LINES))
    assert rules(f'def ok():\n    """Doc."""\n{short}\n') == []


def test_global_state_is_flagged():
    assert rules("counter = 0\ndef f():\n    global counter\n    counter += 1\n") == ["P10-6"]
    assert rules("cache = {}\n") == ["P10-6"]
    assert rules("cache: list[int] = list()\n") == ["P10-6"]
    assert rules("CACHE = {}\n") == []


def test_exec_and_eval_are_flagged():
    assert rules("eval('1 + 1')\nexec('x = 1')\n") == ["P10-8", "P10-8"]


def test_main_reports_and_returns_exit_code(tmp_path: Path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("while True:\n    pass\n", encoding="utf-8")
    good = tmp_path / "good.py"
    good.write_text("X = 1\n", encoding="utf-8")
    assert main([str(good)]) == 0
    assert main([str(tmp_path)]) == 1
    assert "bad.py:1: P10-2" in capsys.readouterr().out


def test_repository_code_follows_the_rules():
    """The project's own code must pass its own standard."""
    assert main(["src", "tools", "app"]) == 0
