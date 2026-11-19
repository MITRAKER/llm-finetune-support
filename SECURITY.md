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
| PYSEC-2026-2447 (CVE-2025-69872) | `diskcache` 5.6.3, a dependency of `llama-cpp-python` in the demo app | No fixed version exists yet. Exploiting it needs write access to the cache folder, and the app never uses llama-cpp-python's disk cache. Remove the exception when a fix is released. |
| PYSEC-2026-3716 (CVE-2026-66007) | `datasets` 4.8.5, in the training stack | Fixed in 5.0.1, which Unsloth 2026.11.5 does not allow (it needs datasets<5). The flaw is in the folder-based dataset builders; the pipeline only loads the Bitext CSV and its own JSONL files. Remove the exception when Unsloth allows datasets 5. |
| PYSEC-2026-4217 (CVE-2026-104851) | `fsspec` 2026.2.0, required by `datasets` 4.8.5 | Fixed in 2026.6.0, which datasets 4.8.5 does not allow. The flaw is in fsspec's `ReferenceFileSystem`, which the pipeline never uses. Remove the exception together with the one above. |

The CI audit ignores only these IDs.

## How this repository protects itself

- **Secrets** stay out of git: they live in Kaggle, GitHub and Hugging Face secret
  stores and are read from environment variables. A gitleaks pre-commit hook and
  GitHub secret scanning catch mistakes.
- **Dependencies** are pinned and audited with pip-audit in CI: the dev tools, the
  CPU and GPU training stacks, the demo app and the Space's Gradio version. Accepted
  advisories are listed above with the reason. Dependabot proposes upgrades as pull
  requests.
- **Code** is scanned by CodeQL and by ruff's security rules (bandit) on every pull request.
- **Workflows** run with read-only permissions by default. The deploy workflow runs
  after CI passes on a push to this repository's `main` branch, or when someone with
  write access starts it by hand on `main`; it refuses to run on any other branch.
