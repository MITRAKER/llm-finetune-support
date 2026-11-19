"""Tests for the Space deployment script, with a fake Hub client (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tools import deploy_space


class FakeApi:
    """Records the calls deploy() makes instead of talking to the Hub."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def create_repo(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("create_repo", args, kwargs))

    def upload_folder(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", args, kwargs))

    def add_space_variable(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("add_space_variable", args, kwargs))


def test_deploy_uploads_the_real_app_folder():
    api = FakeApi()
    url = deploy_space.deploy(api, "someone/support-bot", model_repo="someone/support-bot-GGUF")
    assert url == "https://huggingface.co/spaces/someone/support-bot"
    names = [name for name, _, _ in api.calls]
    assert names == ["create_repo", "add_space_variable", "upload_folder"]  # the variable is set before the restart
    assert api.calls[1][1] == ("someone/support-bot", "MODEL_REPO", "someone/support-bot-GGUF")
    upload = api.calls[2][2]
    assert upload["repo_type"] == "space"
    assert Path(upload["folder_path"]) == deploy_space.APP_DIR


def test_deploy_without_model_repo_skips_the_variable():
    api = FakeApi()
    deploy_space.deploy(api, "someone/support-bot")
    assert [name for name, _, _ in api.calls] == ["create_repo", "upload_folder"]


def test_deploy_validates_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="owner/name"):
        deploy_space.deploy(FakeApi(), "no-slash")
    with pytest.raises(FileNotFoundError, match=r"app\.py"):
        deploy_space.deploy(FakeApi(), "someone/x", app_dir=tmp_path)


def test_space_readme_matches_the_app():
    """The Space card must point at app.py and pin the Gradio version the app was written for."""
    text = (deploy_space.APP_DIR / "README.md").read_text(encoding="utf-8")
    front_matter = text.split("---")[1]
    assert "sdk: gradio" in front_matter
    assert "app_file: app.py" in front_matter
    assert "sdk_version: 6.32.0" in front_matter
