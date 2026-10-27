"""The demo task-corpus generator is valid, deterministic and places matches in BANDS.

Coverage of the per-task hours step depends on distinct template texts; the band a
match lands in depends on how much the neighbours' hours disagree
(``reliability = weighted_similarity × (1 − dispersion)``). This corpus authors that
disagreement per template — ``tight`` (green), ``noisy`` and ``contested`` (Jev's
band) — so the profile → coefficient-of-variation mapping is the invariant tested here.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

from app.generation.rag.schemas import Budget

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_demo_corpus import (  # noqa: E402
    ABBREV,
    COMMERCE_MODULES,
    DOCUMENT_TYPE,
    LEARNING_MODULES,
    PLATFORM_MODULES,
    PROFILE,
    generate_corpus,
)
from build_task_corpus import DOCUMENT_TYPE as BASE_DOCUMENT_TYPE  # noqa: E402

CATALOGS = (COMMERCE_MODULES, LEARNING_MODULES, PLATFORM_MODULES)
ALL_TEMPLATES = [t for catalog in CATALOGS for templates in catalog.values() for t in templates]


def _hours_by_template(corpus):
    by: dict[str, list[int]] = {}
    for project in corpus:
        for c in project["components"]:
            by.setdefault(c["name"], []).append(c["estimated_hours"])
    return by


def _median_cv_of_five(hours: list[int], rng: random.Random) -> float:
    cvs = []
    for _ in range(200):
        sample = rng.sample(hours, 5)
        cvs.append(statistics.pstdev(sample) / statistics.fmean(sample))
    return statistics.median(cvs)


def test_generated_corpus_validates_as_budget():
    corpus = generate_corpus(count=6, seed=3)
    for budget in (Budget.model_validate(p) for p in corpus):
        assert budget.components
        assert all(c.module for c in budget.components)
        assert budget.total_estimated_hours == sum(c.estimated_hours for c in budget.components)


def test_generation_is_deterministic_for_a_seed():
    assert generate_corpus(count=4, seed=11) == generate_corpus(count=4, seed=11)


def test_budget_ids_are_unique_and_namespaced():
    ids = [p["budget_id"] for p in generate_corpus()]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("DEMO-") for i in ids)


def test_both_demo_domains_use_existing_sectors():
    """A new ``Sector`` literal value would 422 on ingest."""
    assert {p["client_metadata"]["sector"] for p in generate_corpus()} == {"ecommerce", "education"}


def test_lands_under_its_own_document_type():
    assert DOCUMENT_TYPE == "demo_task_breakdown" != BASE_DOCUMENT_TYPE


def test_every_module_has_an_abbreviation_and_every_profile_a_template():
    modules = {m for catalog in CATALOGS for m in catalog}
    assert modules <= set(ABBREV)
    names = {name for name, *_ in ALL_TEMPLATES}
    assert set(PROFILE) <= names, set(PROFILE) - names  # no typo'd profile keys
    assert set(PROFILE.values()) <= {"noisy", "contested"}


def test_base_ranges_are_tight():
    """Dispersion comes ONLY from the profile, never from a wide authored range."""
    offenders = [(n, lo, hi) for n, _d, _t, (lo, hi), _c in ALL_TEMPLATES if hi / lo > 1.25]
    assert not offenders


def test_every_template_has_enough_projects_to_fill_top_k():
    by = _hours_by_template(generate_corpus())
    assert min(len(v) for v in by.values()) >= 5


def test_profiles_place_the_consensus_in_the_intended_band():
    by = _hours_by_template(generate_corpus())
    rng = random.Random(0)
    for name, hours in by.items():
        cv = _median_cv_of_five(hours, rng)
        profile = PROFILE.get(name, "tight")
        if profile == "tight":
            assert cv <= 0.1, (name, cv)
        else:  # noisy / contested: CV ≈ 0.4–0.75 → reliability ≈ 0.15–0.35 at sim ≈ 0.6–0.7
            assert 0.25 <= cv <= 0.8, (name, profile, cv)
