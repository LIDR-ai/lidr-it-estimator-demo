from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import (
    get_estimation_service,
    get_llm_wrapper,
    get_openai_client,
    get_session_store,
)
from app.main import app
from app.domain.schemas.estimation import EstimationResult
from app.domain.estimation_service import EstimationService
from app.generation.conversation.models import ProjectMetadata
from app.generation.conversation.store import SessionStore


@pytest.fixture(autouse=True)
def _disable_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the Session 15 service-token middleware off for the whole suite.

    ``Settings`` is configured with ``env_file=".env"``, so running pytest from
    ``ai-service/`` picks up the developer's local ``.env``. If that file carries
    an ``AI_SERVICE_TOKEN`` — which it will on any machine that also runs the
    Docker stack — the middleware would switch on and every ``TestClient`` call
    in the suite would come back 401, since none of them sends the header.

    Pinning it to ``None`` here makes the tests independent of local
    configuration. The middleware's own behaviour is covered explicitly in
    ``tests/api/test_service_token.py``, which sets the value on purpose.
    """
    monkeypatch.setattr(get_settings(), "AI_SERVICE_TOKEN", None)


@pytest.fixture(autouse=True)
def _disable_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real Jev client out of the suite.

    Jev is ON by default (``GRAPH_JEV_ENABLED=True``) and only needs a key to start
    making HTTPS calls, so a local ``.env`` with ``JEV_API_KEY`` would send every graph
    test to jevai.org. Pinning the key to ``None`` makes ``get_jev_classifier()``
    return ``None``; the triage tests inject their own fake classifier on purpose.
    """
    from app.dependencies import get_jev_classifier

    monkeypatch.setattr(get_settings(), "JEV_API_KEY", None)
    get_jev_classifier.cache_clear()


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI test client configured with the application."""
    return TestClient(app)


# ---- Shared fakes for the conversational integration tests ----


def make_canned_result(
    *,
    total_cost_eur: int = 25_000,
    total_duration_weeks: int = 6,
    confidence_pct: int = 72,
) -> EstimationResult:
    return EstimationResult(
        summary="Canned CRM build for the sales team.",
        confidence_pct=confidence_pct,
        phases=[
            {"name": "Discovery", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Scoping workshops + tech spike."},
            {"name": "Build", "duration_weeks": total_duration_weeks - 1,
             "cost_eur": total_cost_eur - 5_000,
             "summary": "Core build with React + Postgres."},
        ],
        total_duration_weeks=total_duration_weeks,
        total_cost_eur=total_cost_eur,
    )


class FakeLLMWrapper:
    """In-process double of ``LLMWrapper`` for conversational tests.

    Captures every ``complete_structured_chat`` call. Returns scripted
    EstimationResult / ProjectMetadata pairs for the estimation+extractor
    sequence. For any other ``response_model`` (e.g. summary envelopes from
    the compressor, critic feedback from the Boss), it returns a canned
    instance produced by the registered factory or a sensible default.

    Tests can register factories with ``register_response_for(schema, factory)``.
    """

    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.scripted: list[tuple[EstimationResult, ProjectMetadata]] = []
        self._turn = 0
        self._extra_factories: dict[type, callable] = {}

    def add_turn(
        self,
        *,
        result: EstimationResult | None = None,
        metadata: ProjectMetadata | None = None,
    ) -> None:
        self.scripted.append(
            (result or make_canned_result(), metadata or ProjectMetadata())
        )

    def register_response_for(self, schema: type, factory) -> None:
        """Register a factory that produces an instance of ``schema``.

        ``factory`` is a zero-arg callable. Useful when a test pipeline
        triggers a third Pydantic call (summarizer, critic) that the default
        estimation/metadata pair-script doesn't cover.
        """
        self._extra_factories[schema] = factory

    def _default_for(self, schema: type):
        """Best-effort canned instance when no factory is registered."""
        # Local imports keep this lazy — the optional schemas are only present
        # once their modules ship.
        from app.generation.conversation.compression.summarizer import _SummaryEnvelope
        from app.generation.conversation.compression.anchors import _AnchorClassification

        if schema is _SummaryEnvelope:
            return _SummaryEnvelope(summary="(canned summary for tests)")
        if schema is _AnchorClassification:
            return _AnchorClassification(is_anchor=False, reason="default")
        # Final fallback: try to construct with no args.
        return schema()

    def complete_structured_chat(self, *, messages, response_model, **kwargs):
        self.chat_calls.append(
            {
                "messages": messages,
                "response_model": response_model.__name__,
                "kwargs": kwargs,
            }
        )
        meta = {"model": "gpt-4o-mini", "provider": "openai", "latency_ms": 1}

        if response_model is EstimationResult:
            idx = self._turn // 2
            if idx >= len(self.scripted):
                self.scripted.append((make_canned_result(), ProjectMetadata()))
            result, _metadata = self.scripted[idx]
            self._turn += 1
            return result, meta

        if response_model is ProjectMetadata:
            idx = self._turn // 2
            if idx >= len(self.scripted):
                self.scripted.append((make_canned_result(), ProjectMetadata()))
            _result, metadata = self.scripted[idx]
            self._turn += 1
            return metadata, meta

        # Third-party schemas (summary envelope, critic feedback, …).
        factory = self._extra_factories.get(response_model)
        if factory is not None:
            return factory(), meta
        return self._default_for(response_model), meta


@pytest.fixture
def fake_wrapper() -> FakeLLMWrapper:
    return FakeLLMWrapper()


@pytest.fixture
def session_store_factory():
    """Factory so a test can pick its own ``max_turns``."""
    def _factory(*, max_turns: int = 6) -> SessionStore:
        return SessionStore(max_turns=max_turns)

    return _factory


@pytest.fixture
def conversational_client(fake_wrapper: FakeLLMWrapper, session_store_factory):
    """Wire FastAPI to use the fake wrapper and a fresh in-memory store."""

    store = session_store_factory()

    service = EstimationService(
        llm_wrapper=fake_wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        metadata_extractor_model="gpt-4o-mini",
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_llm_wrapper] = lambda: fake_wrapper
    app.dependency_overrides[get_openai_client] = lambda: None

    with TestClient(app) as c:
        yield c, store

    app.dependency_overrides.clear()
