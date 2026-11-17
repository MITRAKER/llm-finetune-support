"""Continuous deployment: publish the demo app in app/ to a Hugging Face Space.

The GitHub workflow .github/workflows/deploy-space.yml runs this after every
change to app/ on main, so the public demo always matches the reviewed code.
It can also be run by hand:

    HF_TOKEN=... python tools/deploy_space.py --space your-name/support-bot --model-repo your-name/support-bot-GGUF
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent.parent / "app"
IGNORE = ["__pycache__/*", "*.pyc", ".pytest_cache/*"]


def deploy(api: Any, space_id: str, app_dir: Path = APP_DIR, model_repo: str | None = None) -> str:
    """Create the Space if needed, upload the app and point it at the GGUF model repo."""
    if "/" not in space_id:
        raise ValueError(f"space_id must look like 'owner/name', got {space_id!r}")
    if not (app_dir / "app.py").is_file() or not (app_dir / "README.md").is_file():
        raise FileNotFoundError(f"{app_dir} must contain app.py and the Space README.md")
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(app_dir),
        ignore_patterns=IGNORE,
        commit_message="Deploy the demo app from GitHub",
    )
    if model_repo:
        api.add_space_variable(space_id, "MODEL_REPO", model_repo)  # read by app.py at startup
    return f"https://huggingface.co/spaces/{space_id}"


def main(argv: list[str] | None = None) -> str:  # pragma: no cover - talks to the Hugging Face Hub
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="Space id, e.g. your-name/support-bot")
    parser.add_argument("--model-repo", help="Hub repo with the GGUF file (sets the MODEL_REPO Space variable)")
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set the HF_TOKEN environment variable (a token with write access).")
    url = deploy(HfApi(token=token), args.space, model_repo=args.model_repo)
    print(f"Deployed {url}")
    return url


if __name__ == "__main__":  # pragma: no cover
    main()
