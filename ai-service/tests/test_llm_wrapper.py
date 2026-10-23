from types import SimpleNamespace
from unittest.mock import patch

import fakeredis
import pytest

from app.generation.cag.exact import EstimationCache
from app.foundation.llm.wrapper import LLMWrapper, _estimate_cost


def _fake_completion(model: str, content: str = "the answer", input_tokens: int = 100, output_tokens: int = 50):
    """Build a SimpleNamespace shaped like a litellm.ModelResponse."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


@pytest.fixture
def wrapper() -> LLMWrapper:
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    return LLMWrapper(
        openai_api_key="fake-openai",
        anthropic_api_key="fake-anthropic",
        primary_model="gpt-4o-mini",
        fallback_model="claude-haiku-4-5-20251001",
        timeout=30,
        num_retries=2,
        cache=cache,
    )


def test_estimate_cost_uses_pricing_table() -> None:
    cost = _estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    # 1M input * 0.15 + 1M output * 0.60 = 0.75 USD
    assert cost == pytest.approx(0.75)


def test_complete_returns_normalised_dict_and_caches(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="gpt-4o-mini", content="hello world")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        result = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    assert mocked.call_count == 1
    assert result["estimation"] == "hello world"
    assert result["model"] == "gpt-4o-mini"
    assert result["provider"] == "openai"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 50
    assert result["cache_hit"] is False
    assert result["cost_usd"] > 0

    # Second call with the same inputs should hit the cache without invoking the router.
    with patch.object(wrapper.router, "completion") as mocked_again:
        cached = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    assert mocked_again.call_count == 0
    assert cached["cache_hit"] is True
    assert cached["estimation"] == "hello world"


def test_complete_with_model_override_bypasses_router(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="gpt-4o", content="overridden")
    with patch("app.foundation.llm.wrapper.litellm.completion", return_value=fake) as direct, \
        patch.object(wrapper.router, "completion") as router_call:
        result = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override="gpt-4o",
            max_tokens=4000,
            thinking_budget=None,
        )
    assert direct.call_count == 1
    assert router_call.call_count == 0
    assert direct.call_args.kwargs["model"] == "gpt-4o"
    assert result["model"] == "gpt-4o"


def test_thinking_budget_passed_for_anthropic_fallback(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="claude-haiku-4-5-20251001", content="ok")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=2048,
        )
    # primary is OpenAI (gpt-4o-mini), so thinking budget is *ignored* in kwargs.
    assert "thinking" not in mocked.call_args.kwargs


def test_thinking_budget_pads_max_tokens_when_anthropic_override(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="claude-haiku-4-5-20251001", content="ok")
    with patch("app.foundation.llm.wrapper.litellm.completion", return_value=fake) as direct:
        wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override="claude-haiku-4-5-20251001",
            max_tokens=1000,
            thinking_budget=4096,
        )
    kwargs = direct.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert kwargs["max_tokens"] == 4096 + 1024


def test_complete_structured_chat_forwards_messages(wrapper: LLMWrapper) -> None:
    """Conversational structured call: caller passes the full ``messages``
    list (system + history + current user) and Instructor returns a Pydantic
    model atomically."""
    from pydantic import BaseModel

    class _Answer(BaseModel):
        text: str

    messages = [
        {"role": "system", "content": "you are an estimator"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second user"},
    ]

    expected = _Answer(text="ok")
    with patch.object(
        wrapper._instructor.chat.completions, "create", return_value=expected
    ) as mocked:
        result, meta = wrapper.complete_structured_chat(
            messages=messages,
            response_model=_Answer,
        )

    assert result is expected
    kwargs = mocked.call_args.kwargs
    assert kwargs["messages"] is messages
    assert kwargs["response_model"] is _Answer
    assert kwargs["model"] == "gpt-4o-mini"
    assert meta["model"] == "gpt-4o-mini"
    assert meta["provider"] == "openai"
    assert "latency_ms" in meta


def test_complete_structured_chat_uses_anthropic_key_for_claude(wrapper: LLMWrapper) -> None:
    from pydantic import BaseModel

    class _Answer(BaseModel):
        text: str

    with patch.object(
        wrapper._instructor.chat.completions, "create", return_value=_Answer(text="x")
    ) as mocked:
        wrapper.complete_structured_chat(
            messages=[{"role": "user", "content": "hi"}],
            response_model=_Answer,
            model_override="claude-haiku-4-5-20251001",
        )
    assert mocked.call_args.kwargs["api_key"] == "fake-anthropic"


# test_complete_stream_yields_chunks_and_caches was removed in Session 4 when
# the /api/v1/estimate/stream endpoint and the wrapper's complete_stream() method
# were deleted. Structured output via Instructor (complete_structured) replaces
# token streaming; tests for that path live in test_estimate_endpoint.py with a
# mocked EstimationService.


# --- Runtime model overrides (Settings UI) ----------------------------------


def _wrapper_with_runtime(primary_override: str | None) -> LLMWrapper:
    """Wrapper wired to a runtime config store with an optional primary override."""
    import fakeredis as _fakeredis

    from app.config import Settings
    from app.foundation.llm.runtime_config import RuntimeModelConfig

    settings = Settings(OPENAI_API_KEY="fake-openai", _env_file=None)
    runtime = RuntimeModelConfig(_fakeredis.FakeRedis(decode_responses=True), settings)
    if primary_override:
        runtime.set("PRIMARY_MODEL", primary_override)

    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    return LLMWrapper(
        openai_api_key="fake-openai",
        anthropic_api_key="fake-anthropic",
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=30,
        num_retries=2,
        cache=cache,
        runtime_config=runtime,
    )


def test_primary_model_property_reflects_runtime_override() -> None:
    wrapper = _wrapper_with_runtime("gpt-4o")
    assert wrapper.primary_model == "gpt-4o"
    # Clearing the override falls back to the settings default.
    wrapper._runtime_config.set("PRIMARY_MODEL", None)
    assert wrapper.primary_model == "gpt-4o-mini"


def test_complete_uses_router_when_no_runtime_override() -> None:
    wrapper = _wrapper_with_runtime(None)
    fake = _fake_completion(model="gpt-4o-mini", content="router path")
    with patch.object(wrapper.router, "completion", return_value=fake) as router_call:
        result = wrapper.complete(system_prompt="sys", user_message="usr")
    assert router_call.call_count == 1
    assert result["model"] == "gpt-4o-mini"


def test_complete_takes_direct_path_when_runtime_override_active() -> None:
    # An active primary override behaves like model_override: direct call,
    # Router untouched (its deployments are frozen at construction).
    wrapper = _wrapper_with_runtime("gpt-4o")
    fake = _fake_completion(model="gpt-4o", content="runtime override")
    with patch("app.foundation.llm.wrapper.litellm.completion", return_value=fake) as direct, \
        patch.object(wrapper.router, "completion") as router_call:
        result = wrapper.complete(system_prompt="sys", user_message="usr")
    assert direct.call_count == 1
    assert router_call.call_count == 0
    assert direct.call_args.kwargs["model"] == "gpt-4o"
    assert result["model"] == "gpt-4o"


def test_complete_structured_meters_instructor_raw_usage(wrapper: LLMWrapper) -> None:
    """Instructor's ``_raw_response`` usage lands in ``meta`` and in the active ledger."""
    from types import SimpleNamespace

    from pydantic import BaseModel

    from app.foundation.llm.usage_meter import metered

    class _Answer(BaseModel):
        text: str

    answer = _Answer(text="ok")
    answer._raw_response = SimpleNamespace(  # what Instructor attaches
        usage=SimpleNamespace(prompt_tokens=1_000, completion_tokens=200)
    )
    with metered() as ledger, patch.object(
        wrapper._instructor.chat.completions, "create", return_value=answer
    ):
        _, meta = wrapper.complete_structured(
            system_prompt="s", user_message="u", response_model=_Answer
        )

    assert (meta["input_tokens"], meta["output_tokens"]) == (1_000, 200)
    assert meta["cost_usd"] == _estimate_cost("gpt-4o-mini", 1_000, 200)
    [rec] = ledger.records
    assert (rec.source, rec.model, rec.input_tokens) == ("_Answer", "gpt-4o-mini", 1_000)
