# Security policy

## Reporting a problem

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on this repository
(https://github.com/MITRAKER/llm-finetune-support/security/advisories/new).
Do not open a public issue. You should get an answer within a week.

## Supported versions

Only the latest release on the `main` branch receives fixes.

## What counts

- Secrets: a token or API key committed anywhere in the history.
- The training or deployment code doing something unsafe with your files or credentials.
- The demo app leaking data or running code it should not.
- Dependencies with known vulnerabilities that our automated checks missed.

Model behaviour problems, such as a reply that ignores the system prompt after a
prompt-injection attempt, are welcome as normal issues: the model is a demo and the
evaluation set already includes such prompts.

## Accepted advisories

| Advisory | Package | Why it is accepted |
|---|---|---|
| PYSEC-2026-2447 (CVE-2025-69872) | `diskcache` 5.6.3, a dependency of `llama-cpp-python` in the demo app | No fixed version exists yet. Exploiting it needs write access to the cache folder, and the app never uses llama-cpp-python's disk cache. The CI audit ignores only this ID; remove the exception when a fix is released. |

## How this repository protects itself

- **Secrets** stay out of git: they live in Kaggle, GitHub and Hugging Face secret
  stores and are read from environment variables. A gitleaks pre-commit hook and
  GitHub secret scanning catch mistakes.
- **Dependencies** are pinned and audited with pip-audit in CI; Dependabot proposes
  upgrades as pull requests.
- **Code** is scanned by CodeQL and by ruff's security rules (bandit) on every pull request.
- **Workflows** run with read-only permissions by default. The deploy workflow only
  runs for pushes to this repository's `main` branch after CI passes.
