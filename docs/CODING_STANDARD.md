# Coding standard: NASA/JPL "Power of Ten", adapted to Python

In 2006 Gerard Holzmann of NASA's Jet Propulsion Laboratory wrote ten rules for
safety-critical C code ("The Power of Ten: Rules for Developing Safety Critical
Code", IEEE Computer, June 2006). The idea behind all ten: code should be simple
enough that a tool can check it and a reviewer can understand it.

Python is not C, so a few rules do not apply literally. This page shows how each
rule is translated for this repository and, most importantly, **which tool
enforces it**. A rule that no tool checks is only a wish. pre-commit runs ruff,
mypy and the Power of Ten checker on every commit; CI runs those plus the full
test suite, pip-audit and CodeQL on every pull request. Rules 2, 3 and 9 also rely
on code review, because no tool can check them completely.

| # | Original rule (C) | What it means here | Enforced by |
|---|---|---|---|
| 1 | Simple control flow: no `goto`, `setjmp`/`longjmp` or recursion | No recursion, direct or indirect: the checker finds functions of one module that call each other in a cycle (recursion across modules is left to code review). Low branching: cyclomatic complexity at most 10, at most 12 branches and 6 returns per function | `tools/check_power_of_ten.py` (P10-1), ruff `C901`, `PLR0912`, `PLR0911` |
| 2 | Every loop has a fixed upper bound | No `while True:`. Loops run over finite collections, and anything that could grow has an explicit cap: `MAX_FILES`, `MAX_WORKERS_LIMIT`, `MAX_HISTORY_MESSAGES`, `max_new_tokens`, `max_steps`. Network and subprocess calls have timeouts or bounded retries | `tools/check_power_of_ten.py` (P10-2), code review |
| 3 | No dynamic memory allocation after initialization | Python allocates for us, so the rule becomes "bounded resources": fixed batch sizes and sequence lengths, a bounded chat history, a bounded thread pool, and the model loaded once (`lru_cache`), never per request | Config values, code review, the CPU smoke test |
| 4 | Functions fit on one printed page (about 60 lines) | At most 60 lines and 50 statements per function, at most 8 arguments | `tools/check_power_of_ten.py` (P10-4), ruff `PLR0915`, `PLR0913` |
| 5 | Assertion density of at least two per function on average; assertions have no side effects, and a failing one triggers an explicit recovery action | Python's `assert` disappears under `python -O`, so production code checks its inputs with explicit exceptions that say how to recover ("Run step 1 first"). Tests are where `assert` lives. The quality gates are assertions about the model itself: a failing gate stops the pipeline with exit code 3 | ruff `S101` (no `assert` outside tests), the tests, `evaluate.py --compare` |
| 6 | Declare data at the smallest possible scope | No `global` or `nonlocal`, and no module-level lists, dicts or sets with lower-case names. UPPER_CASE names are constants by convention: the checker trusts the name and does not check that the value never changes, so never mutate one. State is passed explicitly as arguments | `tools/check_power_of_ten.py` (P10-6) |
| 7 | Check every return value and every parameter | Full type hints checked by mypy (no untyped functions, no implicit `Any` returns). Functions validate their arguments at the boundary. `subprocess.run(check=True)`. Judge API errors are recorded per item, never silently dropped | mypy (`disallow_untyped_defs`, `warn_return_any`, ...), ruff `B`, `RET` |
| 8 | Limit the preprocessor | Python's equivalent of macros is code that writes code: no `exec`, `eval` or `__import__` | `tools/check_power_of_ten.py` (P10-8), ruff `S102`, `S307` |
| 9 | Restrict pointers: one level of dereferencing, no function pointers | Keep indirection shallow: plain data classes and dicts. Callables are passed only as typed test seams (for example `judge_fn` in `judge.py`), never as hidden control flow | mypy (`Callable[...]` types), code review |
| 10 | Compile with all warnings on; zero warnings; static analysis every day | Every warning is an error in the test suite. ruff (style, bugs, security), mypy, CodeQL, pip-audit and the Power of Ten checker run on every pull request, and CodeQL also runs weekly | `pytest` `filterwarnings = error`, `.github/workflows/ci.yml`, `.github/workflows/codeql.yml` |

## Run the checks

```bash
make check            # lint, format, types, Power of Ten and every test (CI also runs pip-audit, Python 3.11 to 3.13 and CodeQL)
make check-fast       # the same without the CPU training run, which needs Linux
make power-of-ten     # only the Power of Ten checker
```

The checker prints one line per violation (`path:line: rule message`) and exits
with code 1 if there are any, so it can block a commit or a merge.

## When a rule is wrong for a case

Rules exist to make code easier to trust, not to win arguments. If a rule blocks
a clearly better design, change the code or the rule in a pull request and say
why. Do not silence a check silently: the only `noqa` comments in the code each
say why on the same line (calling `git` from `PATH`, and importing `unsloth`
first so it can patch the other libraries).
