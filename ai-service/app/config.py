from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Session 2 fields (kept for backwards compatibility with the live demos) ---
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: Literal["openai", "anthropic"] = "anthropic"
    LLM_MODEL: str = "claude-haiku-4-5"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"

    # --- Session 3 fields (LiteLLM wrapper, Redis cache, Streamlit transport) ---
    PRIMARY_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2
    # Catalog of models selectable at runtime via PUT /api/v1/config/models
    # (kept aligned with MODEL_COSTS in app/foundation/llm/wrapper.py). The
    # endpoint filters this list by the API keys actually configured.
    AVAILABLE_MODELS: list[str] = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5",
        "gpt-5-mini",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5",
    ]

    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400

    # --- Session 4 fields (semantic cache) ---
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_TTL: int = 86400
    # When True, the semantic cache LOGS potential hits but does NOT serve them.
    # Used to gather metrics before flipping the cache on in production.
    SEMANTIC_CACHE_LOG_ONLY: bool = False

    ESTIMATOR_API_BASE_URL: str = "http://localhost:8000"

    # --- Session 5 fields (conversational memory + attachments) ---
    # MAX_CONVERSATION_TURNS counts user+assistant pairs. The system prompt is
    # always preserved as an invariant on top of the window.
    MAX_CONVERSATION_TURNS: int = 6
    # Hard cap per extracted attachment (in characters) to protect the context
    # window. Real chunking enters in module 3.
    MAX_ATTACHMENT_CHARS: int = 60_000
    # The metadata extractor runs once per turn; a small/cheap model is enough.
    METADATA_EXTRACTOR_MODEL: str = "gpt-4o-mini"

    # --- Session 5 live: compression + tier + ACB ---
    # Anchor detector: "heuristic" (regex over key phrases) or "llm" (binary
    # classifier via Instructor). Heuristic is the default for cost.
    ANCHOR_DETECTION_MODE: Literal["heuristic", "llm"] = "heuristic"
    # Cheap model used by the cumulative summarizer (history compression).
    COMPRESSION_MODEL: str = "gpt-4o-mini"
    # Conversational prompt version used by ``estimate_conversational``.
    # v2 = pre-live-session baseline. v3 = adds <audience> block driven by tier
    # and an optional <critic_feedback> block consumed by the Boss.
    CONVERSATIONAL_PROMPT_VERSION: str = "v3"
    # Critic model (read-only auditor; cheap is fine).
    CRITIC_MODEL: str = "gpt-4o-mini"
    # Max iterations the Boss can drive (each iteration = 1 actor + 1 critic call).
    # Three is the practical floor: one initial draft + two directed retries.
    # With only two iterations the actor often cannot address all flagged issues
    # in the single available retry, and the loop falls back without converging.
    BOSS_MAX_ITERATIONS: int = 3

    # --- Session 6 fields (data-driven AI: persistence + ingestion + PII) ---
    # Postgres connection string. pgvector/pgvector:pg16 image; the extension
    # is dormant in S06 (no CREATE EXTENSION vector) and only activates in S07.
    DATABASE_URL: str = "postgresql+psycopg://estimator:estimator@localhost:5433/estimator"
    # Where the YAML catalog lives. Resolved relative to the working directory.
    CATALOG_PATH: Path = Path("data/catalog/catalog.yaml")
    # Root where ``CatalogSource.location`` entries are resolved against.
    INGESTION_DATA_ROOT: Path = Path("data/seed")
    # spaCy model loaded by the Presidio AnalyzerEngine. Must be the Spanish
    # one for the live session; ``es_core_news_md`` is the recommended size.
    PRESIDIO_SPACY_MODEL: str = "es_core_news_md"
    # Locale used by Faker to generate consistent pseudonyms per entity_type.
    PSEUDONYM_FAKER_LOCALE: str = "es_ES"
    # HMAC salt. Stored in env so it can be rotated independently of the code.
    PSEUDONYM_HASH_SALT: str = "change-me-in-prod"

    # --- Session 7 live fields (chunking strategies that call external APIs) ---
    # LLM that decomposes a component into atomic propositions (one call per
    # component). A small/cheap model is enough.
    PROPOSITIONAL_CHUNKER_MODEL: str = "gpt-4o-mini"
    # Claude model used by Contextual Retrieval to situate each chunk inside its
    # parent budget. Prompt caching makes the (large) parent document cheap to
    # reuse across the chunks of the same budget.
    CONTEXTUAL_CHUNKER_MODEL: str = "claude-sonnet-4-5"

    # --- Session 9 fields (RAG estimation: transcript → grounded estimate) ---
    # Query understanding distills a transcript into an EstimationQuery; a small
    # model is enough. Generation reasons over retrieved budgets, so it uses the
    # strongest model with medium reasoning effort. Both go through LLMWrapper.
    REFORMULATION_MODEL: str = "gpt-5-mini"
    GENERATION_MODEL: str = "gpt-5"
    # "high" drives a deeper, more consistent module→task decomposition (the S09
    # article used "medium"; we raise it for the granular modular breakdown).
    GENERATION_REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "high"
    # Token ceiling (reasoning + output) for the RAG structured calls. gpt-5 is a
    # reasoning model: its reasoning tokens count against this budget, so the
    # 4000 wrapper default leaves nothing for the JSON and the call truncates
    # (finish_reason='length'). Generous headroom so high-effort reasoning can
    # finish AND emit the larger nested (modules→tasks) Estimate. It is a CAP,
    # not a target — the model only spends what it needs, so a high value adds no
    # latency on its own.
    GENERATION_MAX_TOKENS: int = 64000
    # Retrieval knobs (locked defaults from the Session 9 articles).
    RETRIEVAL_TOP_K: int = 10
    RETRIEVAL_DISTANCE_THRESHOLD: float = 0.6
    # Token budget for the assembled <source> context block (tiktoken cl100k_base).
    MAX_CONTEXT_TOKENS: int = 16384
    # Idempotency cache for POST /v1/estimate/from-transcript (seconds; 24h).
    IDEMPOTENCY_TTL: int = 86400
    # API keys for the two Session 9 routers. None disables the router (401 on
    # every request) — set them in .env to enable the endpoints.
    RETRIEVAL_API_KEY: str | None = None
    ESTIMATE_API_KEY: str | None = None

    # --- Session 10 fields (hybrid search + cross-encoder reranking) ---
    # Default retrieval mode. "vector" reproduces the Session 9 baseline; "hybrid"
    # fuses the dense and lexical (full-text) branches with RRF. Switchable per
    # request (RetrievalRequest.search_mode) and at runtime (RuntimeRetrievalConfig).
    RETRIEVAL_SEARCH_MODE: Literal["vector", "hybrid"] = "vector"
    # Whether the cross-encoder reranks by default. Off keeps the baseline cheap;
    # the recall-then-rerank path turns on per request / at runtime.
    RERANKER_ENABLED: bool = False
    # Multilingual cross-encoder (ES+EN), small enough for CPU at teaching latency.
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Recall width before reranking/fusion (recall-then-rerank): retrieve this many
    # candidates cheaply, then the cross-encoder rescores them down to RERANK_TOP_N.
    RETRIEVAL_RECALL_TOP_K: int = 50
    RERANK_TOP_N: int = 5
    # RRF smoothing constant (Cormack et al. default). Larger = a document must
    # rank well in BOTH branches to win; smaller = a single #1 can dominate.
    RRF_K: int = 60

    # --- Session 10 live fields (advanced retrieval: multi-index pipeline) ------
    # Each advanced-retrieval stage is independently switchable so it can be
    # measured in isolation (the full pipeline is the MAX path, not the only one).
    # These are the .env defaults; routing/transform/decay also flip at runtime
    # (RuntimeRetrievalConfig → Ajustes UI). Search mode + reranking reuse the
    # existing RETRIEVAL_SEARCH_MODE / RERANKER_ENABLED toggles above.
    RETRIEVAL_ROUTING_ENABLED: bool = True
    QUERY_TRANSFORM_ENABLED: bool = True
    # Soft re-weight; off by default — turn on only with evidence (Article 6's
    # warning against magic-number boosts).
    TEMPORAL_DECAY_ENABLED: bool = False
    # Small, fast models for the router classifier and the query transformer
    # (both in AVAILABLE_MODELS, so switchable in the Ajustes tab). Non-reasoning
    # models on purpose: cheap and no reasoning-token budget to starve the JSON.
    ROUTER_MODEL: str = "gpt-4o-mini"
    QUERY_TRANSFORM_MODEL: str = "gpt-4o-mini"
    # Exponential half-life for temporal decay (weight = 0.5 ** (age/half_life)).
    # ≈2.5 years: budgets age slowly, so recency only breaks ties.
    TEMPORAL_DECAY_HALF_LIFE_DAYS: int = 900
    # Caps for the query transformer (sub-queries) and the router (targets).
    QUERY_MAX_SUBQUERIES: int = 4
    ROUTER_MAX_TARGETS: int = 3

    # --- Session 10 live fields (per-task hours estimation) ---------------------
    # The structure-only generation leaves tasks without hours; each task is then
    # matched against the historical task corpus (chunk_type 'historical_task') and
    # the hours come from a weighted consensus of the nearest neighbours. These two
    # knobs change mid-session (calibrating the red threshold against the corpus),
    # so they flip at runtime via RuntimeRetrievalConfig → Ajustes UI.
    TASK_HOURS_TOP_K: int = 5
    # Cosine-distance floor: a task whose nearest historical task is farther than
    # this gets NO hours (red flag in the UI) instead of a low-confidence guess.
    TASK_HOURS_DISTANCE_THRESHOLD: float = 0.45

    # --- Session 11 live fields (generation quality: hallucination gate) --------
    # A SEMANTIC layer on top of the referential citation check (verify_citations):
    # verify_citations proves every cited chunk_id was retrieved; the gate proves
    # the number is ENTAILED by that chunk. A deterministic numeric anchor + a
    # strict LLM judge grade each grounded line grounded / insufficient / degraded.
    # Switchable at runtime (RuntimeRetrievalConfig → Ajustes UI).
    HALLUCINATION_GATE_ENABLED: bool = True
    # The strict judge that checks a line's cited evidence entails its number. A
    # cheap model is enough; in AVAILABLE_MODELS, so switchable in the Ajustes tab.
    HALLUCINATION_JUDGE_MODEL: str = "gpt-5-mini"
    # Relative tolerance for the numeric anchor: a grounded line whose hours deviate
    # from the historical anchor by more than this fraction is degraded (0.5 = ±50%).
    HALLUCINATION_NUMERIC_TOLERANCE: float = 0.5

    # --- Session 11 live fields (augmentation + synthesis) ----------------------
    # Input-quality layer applied to the retrieved chunks BEFORE generation:
    # compress each source to its key points and reorder with edge-loading
    # (most-relevant first AND last) against lost-in-the-middle. Both switchable
    # at runtime; the reorder is a pure, free transform, the compression optional.
    AUGMENTATION_ENABLED: bool = True
    AUGMENTATION_COMPRESS: bool = True
    AUGMENTATION_REORDER: bool = True
    # Cheap model for the optional LLM compression (extractive compression needs
    # none). In AVAILABLE_MODELS.
    AUGMENTATION_MODEL: str = "gpt-5-mini"
    # Two-stage synthesis of the per-task hours: a deterministic anchor + model
    # judgement. When the historical sources disagree beyond this dispersion
    # (coefficient of variation, e.g. one says 40h and another 90h), emit an hour
    # RANGE with a reason instead of a single point. Switchable at runtime.
    SYNTHESIS_ENABLED: bool = True
    SYNTHESIS_CONTRADICTION_THRESHOLD: float = 0.35

    # --- Session 12 fields (hand-written agentic layer) -------------------------
    # The agent drives a MANUAL tool loop over the raw OpenAI Responses API
    # (client.responses.create) — the one deliberate exception to the "everything
    # goes through LLMWrapper" rule, because the whole point of the exercise is to
    # see the reason→act→observe loop by hand. These are plain defaults; the demo
    # script (scripts/run_agent_s12.py) overrides them per invocation via CLI flags
    # (cheap gpt-5-mini for loop debugging, gpt-5 for the real run). No runtime
    # config: there is no live endpoint this session, only the script.
    AGENT_MODEL: str = "gpt-5"
    AGENT_REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "medium"
    # Safeguard against a runaway loop, on top of the natural stop (a turn with no
    # more tool calls). One iteration == one Responses API round-trip.
    AGENT_MAX_ITERATIONS: int = 10
    # Retrieval knobs the search_budgets tool passes to retrieve(). Looser than the
    # RAG defaults on purpose: the agent issues many narrow per-component queries and
    # benefits from a few more candidates each.
    AGENT_SEARCH_TOP_K: int = 5
    AGENT_SEARCH_DISTANCE_THRESHOLD: float = 0.6

    # --- Session 13 fields (LangGraph estimation graph + observability) ---------
    # The graph re-expresses the estimation flow as an explicit StateGraph inside
    # the service (app/domain/graph). Its structured-output nodes go through
    # LLMWrapper; small non-reasoning models keep the sequential demo cheap and
    # fast. Both are in AVAILABLE_MODELS, so switchable in the Ajustes tab.
    # Extraction/classification are simple structured calls a small model handles;
    # consolidation must reliably populate numeric fields (gpt-4o-mini tends to
    # leave engineer_days null), so it defaults to the stronger gpt-4o.
    GRAPH_EXTRACTION_MODEL: str = "gpt-4o-mini"
    GRAPH_GENERATION_MODEL: str = "gpt-4o"

    # --- Session 13 (live) — the multi-agent orchestration ------------------- #
    # The graph grows from the pre-exercise 5-node pipeline into a pipeline of
    # SPECIALISED AGENTS with explicit handovers and two human gates. Each agent
    # node uses a model sized to its job; the two gpt-5 agents (structure,
    # hours-recovery) reuse AGENT_MODEL / AGENT_* below.
    GRAPH_CLASSIFIER_MODEL: str = "gpt-4o-mini"  # complexity + reformulation (cheap)
    GRAPH_ANALYSIS_MODEL: str = "gpt-4o"  # reliability report needs the stronger model
    GRAPH_PROPOSAL_MODEL: str = "gpt-4o"  # commercial proposal (bonus)
    # The bonus proposal_agent is opt-out via config: flip to false to end the graph
    # right after the final human gate (no proposal drafted).
    GRAPH_PROPOSAL_ENABLED: bool = True
    # Session 13 live (didactic): each graph agent is "played" by a Matrix character
    # whose short persona is prepended to its system prompt. Flip to false to run the
    # agents plain (personas never change the required output shape; see graph/personas.py).
    GRAPH_PERSONAS_ENABLED: bool = True
    # Jev (TypeSafe AI) triage in the recovery join: before Trinity's gpt-5 loop, each
    # doubtful task that HAS hours and reliability >= GRAPH_JEV_MIN_RELIABILITY is sent
    # to Jev, which accepts it as is or escalates it. ON by default, but inert while
    # JEV_API_KEY is blank (no key → no client → the pre-Jev behaviour), so local runs
    # and CI never need Jev. Any Jev failure escalates (the pre-Jev behaviour).
    GRAPH_JEV_ENABLED: bool = True
    # TypeSafe AI's official API (POST /v1/systemone); keys from console.typesafe.ai/keys.
    JEV_API_BASE_URL: str = "https://api.typesafe.ai"
    JEV_API_KEY: str | None = None
    JEV_MODEL: str = "jev-latest"  # alias; GET /v1/models lists what the account can use
    JEV_TIMEOUT: float = 30.0
    JEV_MAX_CONCURRENCY: int = 8
    # Retries on 408/429/5xx (529 = overloaded) and timeouts/connection errors, with
    # exponential backoff — the official SDK's default. Contract errors (400/401/422)
    # are never retried. After them, the batch's circuit opens and the rest are not sent.
    JEV_RETRIES: int = 2
    # What a task gets when Jev cannot answer: "escalate" (fail safe → Trinity) or
    # "simulated" — a deterministic heuristic stand-in, ALWAYS labelled "Jev (simulado)"
    # in the UI/CLI. For demos while the real API is unavailable; never silent.
    JEV_FALLBACK: Literal["escalate", "simulated"] = "escalate"
    # Below this the analog is too weak to be worth asking: straight to Trinity.
    GRAPH_JEV_MIN_RELIABILITY: float = 0.15
    # Jev's probabilities are signals, not authorization — WE set the accept bar.
    GRAPH_JEV_ACCEPT_CONFIDENCE: float = 0.7
    # Trinity (the gpt-5 recovery loop) inside the S13 graph — demo-sized on purpose,
    # independent of AGENT_MAX_ITERATIONS (shared with the S12 endpoint). 3 is the
    # smallest useful cap: turn 1 searches, turn 2 derives the hours, turn 3 stops —
    # with 2 the cap cuts before any derive_task_hours runs. MAX_TASKS hands Trinity
    # only the N worst doubtful tasks (no analog first, then lowest reliability); the
    # rest stay flagged as unrecovered. 0 = no limit.
    GRAPH_RECOVERY_MAX_ITERATIONS: int = 3
    GRAPH_RECOVERY_MAX_TASKS: int = 10
    # classifier's complexity → structure_agent reasoning effort. A richer transcript
    # gets more thinking budget; a simple one stays cheap. Kept as a plain dict so the
    # mapping is data, not a code branch.
    GRAPH_STRUCTURE_EFFORT_BY_COMPLEXITY: dict[str, str] = {
        "low": "low",
        "medium": "medium",
        "high": "high",
    }

    # Logfire service name for the traces. The token itself is read by Logfire from
    # the environment (LOGFIRE_TOKEN) — a run with no token executes every span
    # locally but exports nothing, so observability never breaks startup.
    LOGFIRE_SERVICE_NAME: str = "estimator"

    # --- Session 14: the supervisor multi-agent flow ------------------------ #
    # The supervisor is LLM-DRIVEN (the model owns the control flow) but the decision
    # space is five Literal options over a short factual digest, so a cheap
    # non-reasoning model is the right tool. Already in AVAILABLE_MODELS.
    SUPERVISOR_ROUTER_MODEL: str = "gpt-4o-mini"
    # Hard ceiling on routing steps — the bound that makes an LLM router safe in a
    # graph with cyclic return edges. 8 = the four agents plus room for one legitimate
    # re-route, then finish.
    SUPERVISOR_MAX_STEPS: int = 8
    # Human-review trigger 1: pause below this 0..1 confidence. THE knob of the
    # exercise; raise it to send more estimates to a person.
    SUPERVISOR_CONFIDENCE_THRESHOLD: float = 0.6
    # Human-review trigger 3: pause when fewer than this fraction of components have
    # any precedent at all in the historical budgets.
    SUPERVISOR_MIN_GROUNDED_RATIO: float = 0.5
    # Level 3: false = a denied tool call returns a denial envelope the agent survives;
    # true = it raises PrivilegeViolation and fails the run loudly. False is the
    # teaching default — the run completes AND the denial is visible in the trace.
    SUPERVISOR_PRIVILEGE_STRICT: bool = False
    # How much of a tool's arguments the audit log echoes. The SHA-256 digest is always
    # logged in full, so a call's identity is provable without dumping a transcript.
    SUPERVISOR_AUDIT_ARGS_PREVIEW_CHARS: int = 200

    # --- Session 14 (LIVE) -------------------------------------------------- #
    # Competition pattern: when true, the estimate step runs a conservative-vs-aggressive
    # competition + a synthesizer instead of a single consolidation. Off by default so
    # the reference graph, its router and its tests are byte-for-byte unchanged; the live
    # demo flips it on (build flag / --compete).
    SUPERVISOR_COMPETITION_ENABLED: bool = False
    # How hard the divergence between the two estimators pulls the deterministic
    # confidence down. A wide spread is structural uncertainty the grounding cannot see,
    # so it must be able to trip the HITL gate on its own. Scaled 0..1 by the divergence
    # ratio: penalty = SUPERVISOR_DIVERGENCE_PENALTY * ratio.
    SUPERVISOR_DIVERGENCE_PENALTY: float = 0.4
    # Sandboxing: when true, an irreversible write (save_estimate) is queued after
    # validation, which forces the human gate to pause (the pause authorizes the write)
    # and appends a persistence_agent that executes the write under guard_action. Off by
    # default so the reference flow ends at the gate.
    SUPERVISOR_PERSISTENCE_ENABLED: bool = False

    # --- Session 15 (production architecture & deployment) ------------------- #
    # Shared secret for service-to-service calls (business backend → AI service),
    # carried in the ``X-Service-Token`` header and checked by the middleware in
    # ``app/api/service_token.py`` on every route except /health and the docs.
    #
    # This is the OUTER layer: "are you a service allowed to talk to me at all?".
    # The Session 9 ``RETRIEVAL_API_KEY`` / ``ESTIMATE_API_KEY`` remain the inner
    # layer, answering the finer "which endpoints may this caller use?".
    #
    # Empty/None deliberately DISABLES the check, unlike the S9 keys (where an
    # unset key means 401 on everything). Rationale: this knob sits in front of
    # the whole app, so defaulting it to "on" would break every test and every
    # local `uv run uvicorn` the moment it shipped. Deployments set it; local
    # development and the test suite leave it blank.
    AI_SERVICE_TOKEN: str | None = None

    @model_validator(mode="after")
    def validate_at_least_one_api_key(self) -> "Settings":
        """LiteLLM may try either provider via fallback, so we require at least one key."""
        if not self.OPENAI_API_KEY and not self.ANTHROPIC_API_KEY:
            raise ValueError("At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
