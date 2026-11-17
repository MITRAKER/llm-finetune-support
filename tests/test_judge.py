"""Tests for the LLM judge, using a fake client (no network, no API key)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import judge


def verdict(**overrides: Any) -> judge.JudgeVerdict:
    data = {
        "reason": "fine",
        "correctness": 4,
        "helpfulness": 5,
        "tone": 5,
        "invented_details": False,
        "acceptable": True,
    }
    data.update(overrides)
    return judge.JudgeVerdict(**data)


class FakeMessages:
    """Mimics client.beta.messages.parse and records the requests it receives."""

    def __init__(self, stop_reason: str = "end_turn", parsed: judge.JudgeVerdict | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stop_reason = stop_reason
        self.parsed = parsed if parsed is not None else verdict()

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            parsed_output=self.parsed if self.stop_reason != "refusal" else None,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )


def fake_client(messages: FakeMessages) -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def item(**kw: Any) -> judge.JudgeItem:
    base = {"item_id": "t-1", "user_message": "cancel my order", "reply": "Sure, {{Order Number}}"}
    base.update(kw)
    return judge.JudgeItem(**base)


def test_render_includes_reference_or_expected_behaviour():
    text = item(reference="ref answer").render()
    assert "Reference answer" in text
    assert "ref answer" in text
    assert "Expected behaviour" not in text
    text = item(expected="decline politely").render()
    assert "Expected behaviour:\ndecline politely" in text


def test_validate_verdict_clamps_scores():
    data = judge.validate_verdict(verdict(correctness=9, helpfulness=0, tone=3))
    assert (data["correctness"], data["helpfulness"], data["tone"]) == (5, 1, 3)


def test_request_uses_fallbacks_except_for_haiku():
    opus = judge._request_kwargs("claude-opus-5-5", "low")
    assert opus["fallbacks"] == "default"
    assert opus["betas"] == ["server-side-fallback-2026-07-01"]
    assert opus["output_config"] == {"effort": "low"}
    haiku = judge._request_kwargs("claude-haiku-5-5", "low")
    assert "fallbacks" not in haiku
    assert "betas" not in haiku


def test_judge_one_success_sends_rubric_and_structured_format():
    messages = FakeMessages()
    result = judge.judge_one(fake_client(messages), item(reference="r"), "claude-opus-5-5", "low")
    assert result.error is None
    assert result.verdict is not None
    assert result.verdict["correctness"] == 4
    assert (result.input_tokens, result.output_tokens) == (1000, 200)
    call = messages.calls[0]
    assert call["system"] == judge.RUBRIC
    assert call["output_format"] is judge.JudgeVerdict
    assert call["model"] == "claude-opus-5-5"


def test_judge_one_records_refusal_instead_of_raising():
    result = judge.judge_one(fake_client(FakeMessages(stop_reason="refusal")), item(), "claude-opus-5-5", "low")
    assert result.verdict is None
    assert "refusal" in (result.error or "")


def test_judge_one_records_api_errors():
    anthropic = pytest.importorskip("anthropic")

    class Boom:
        def parse(self, **_: Any) -> None:
            raise anthropic.APIConnectionError(request=SimpleNamespace(method="POST", url="https://x"))

    result = judge.judge_one(fake_client(Boom()), item(), "claude-opus-5-5", "low")  # type: ignore[arg-type]
    assert result.verdict is None
    assert result.error is not None
    assert result.error.startswith("APIConnectionError")


def test_judge_all_and_summary_with_cost_estimate():
    items = [item(item_id=f"t-{i}") for i in range(5)]
    results, summary = judge.judge_all(items, "claude-opus-5-5", client=fake_client(FakeMessages()), max_workers=3)
    assert [r.item_id for r in results] == [f"t-{i}" for i in range(5)]
    assert summary.n == 5
    assert summary.n_errors == 0
    assert summary.scores["judge_correctness"] == 4
    assert summary.scores["judge_acceptable_rate"] == 1.0
    # 5 x (1000 in, 200 out) tokens at $4 / $20 per million tokens
    assert summary.est_cost_usd == pytest.approx(5 * (1000 * 4 + 200 * 20) / 1e6)


def test_summary_of_failures_and_unknown_model_price():
    failed = [judge.JudgeResult("a", None, "boom")]
    summary = judge.summarize(failed, "some-other-model")
    assert summary.n_errors == 1
    assert summary.scores == {}
    assert summary.est_cost_usd is None


def test_judge_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert not judge.judge_available()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert judge.judge_available()
