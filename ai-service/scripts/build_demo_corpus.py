#!/usr/bin/env python3
"""Generate and ingest the DEMO task corpus, additively (Jev triage coverage).

Why this exists. The per-task hours step (``app/generation/rag/task_hours.py``) scores
each match as ``reliability = weighted_similarity × (1 − dispersion)``. With the base
corpus that score is bimodal: a task either finds a tight template (green, nobody
decides) or finds nothing under the distance threshold (straight to Trinity). Jev only
decides the band in between — doubtful rows that HAVE hours (``0.15 ≤ reliability <
0.35``) — so against the real corpus it never had a single candidate.

This corpus covers the task families the two demo transcripts
(``examples/transcripts/05_prueba_tienda_gourmet.txt`` and
``06_demo_academia_corporativa.txt``) left without analog, and authors the hours
DISPERSION per template, because dispersion is the lever that places a match in a band:

* ``tight``     — ``hi/lo <= 1.25``: the analogs agree → green, accepted with no one asked.
* ``noisy``     — a tight consensus plus ~30% outlier projects (×2.5–3): mostly agreeing
                  close analogs → Jev's band, where Jev tends to ACCEPT.
* ``contested`` — bimodal small-scope vs large-scope projects: analogs that disagree →
                  Jev's band, where Jev tends to ESCALATE to Trinity.

A few pieces are deliberately left WITHOUT a template (the habilitation rule engine,
the perishable delivery-day rule), so Trinity keeps real work and the demo shows the
three paths. It is synthetic history, like the base corpus, sized for the demo.

It lands under its own ``document_type`` (droppable on its own) and keeps
``chunk_type='historical_task'`` so the hours search picks it up with no code change.

Usage (inside the container)::

    docker compose exec ai-service python scripts/build_demo_corpus.py --generate-only
    docker compose exec ai-service python scripts/build_demo_corpus.py --ingest

Drop it again — one statement, ``ON DELETE CASCADE`` takes the chunks::

    DELETE FROM documents WHERE document_type = 'demo_task_breakdown';
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_task_corpus import Template, ingest_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "demo_task_corpus.json"
DOCUMENT_TYPE = "demo_task_breakdown"
CHUNK_TYPE = "historical_task"
SOURCE_PREFIX = "data/demo_task_corpus.json"
DEFAULT_SEED = 13
DEFAULT_COUNT = 40

# Hour profiles (see the module docstring). Which occurrences are outliers is fixed by
# the template's occurrence index, not drawn at random: a random draw left some
# templates with no outlier at all (green, not Jev's band). Factors are chosen so the
# coefficient of variation of five neighbours lands in Jev's band, not just "wide".
NOISY_OUTLIER_EVERY = 3  # 1 in 3 projects scoped much larger
NOISY_FACTOR = (2.5, 3.0)
CONTESTED_LARGE_EVERY = 2  # small-scope and large-scope projects alternate
CONTESTED_FACTOR = (3.0, 3.6)

# ---------------------------------------------------------------------------
# Domain A — commerce with subscriptions, stock sync with a legacy POS, returns.
# ---------------------------------------------------------------------------

COMMERCE_MODULES: dict[str, list[Template]] = {
    "Product Catalog & Pricing": [
        (
            "Product detail pages",
            "Product detail page with gallery, origin, allergens and pairing hints "
            "(ficha de producto, alérgenos, origen).",
            ["typescript", "react"],
            (28, 34),
            "medium",
        ),
        (
            "Weight-based pricing & fractional quantities",
            "Price per kilo and fractional quantities for cut-to-order products, with rounding "
            "rules in cart and invoice (precio por peso, al corte, cantidades fraccionadas).",
            ["python", "postgresql"],
            (30, 36),
            "high",
        ),
        (
            "Stock availability messaging",
            "Per-product availability, low-stock and back-in-stock messaging driven by synced "
            "inventory (disponibilidad, aviso de stock).",
            ["typescript", "react"],
            (16, 20),
            "low",
        ),
    ],
    "Search & Recommendations": [
        (
            "Search synonyms, typos & no-results analytics",
            "Synonym dictionaries, typo tolerance and logging of zero-result queries to tune "
            "the catalog search (sinónimos, búsquedas sin resultados).",
            ["python", "elasticsearch"],
            (20, 24),
            "medium",
        ),
        (
            "Food & wine pairing recommendations",
            "Knowledge base of product pairings and a query layer that answers 'cheese for a "
            "red Rioja' style searches with ranked products (maridaje, recomendaciones).",
            ["python", "elasticsearch"],
            (40, 48),
            "high",
        ),
    ],
    "Checkout & Taxes": [
        (
            "VAT calculation & invoice data capture",
            "Spanish VAT rates per product family, tax-inclusive totals and capture of invoice "
            "details at checkout (IVA, datos de facturación, NIF).",
            ["python", "postgresql"],
            (22, 26),
            "medium",
        ),
        (
            "Delivery options & delivery-day selector",
            "Checkout step offering carriers, delivery slots and allowed delivery days computed "
            "from the cart contents (días de entrega, selector de envío).",
            ["typescript", "python"],
            (26, 32),
            "medium",
        ),
        (
            "Totals, rounding & edge-case handling",
            "Order totals with promotions, weighted items and taxes, with rounding and edge "
            "cases covered by tests (redondeos, totales).",
            ["python"],
            (14, 17),
            "medium",
        ),
    ],
    "Subscriptions": [
        (
            "Subscription plan & product model",
            "Monthly subscription box plans, pricing and eligible products "
            "(suscripción mensual, caja sorpresa, plan).",
            ["python", "postgresql"],
            (22, 27),
            "medium",
        ),
        (
            "Subscription lifecycle: pause, skip & cancel",
            "Subscriber self-service to activate, pause, skip a month or cancel, with billing "
            "and cut-off rules (pausar, saltar un mes, baja).",
            ["python", "postgresql"],
            (30, 36),
            "high",
        ),
        (
            "Recurring billing & payment retries",
            "Monthly recurring charges on tokenised cards with dunning and payment retries "
            "(cobro recurrente, reintentos de pago).",
            ["python", "stripe"],
            (32, 38),
            "high",
        ),
        (
            "Box curation & cycle management",
            "Monthly box lineup curation with seasonal products, substitutions and cycle "
            "calendar (selección de productos de temporada, ciclo).",
            ["python", "postgresql"],
            (24, 30),
            "medium",
        ),
        (
            "Batch order generation for subscriptions",
            "Monthly batch job that turns active subscriptions into fulfilment orders with "
            "inventory reservation (generación de pedidos por lote, reserva de stock).",
            ["python", "celery"],
            (26, 32),
            "high",
        ),
        (
            "Subscriber portal",
            "Customer area to manage the subscription, address, payment method and history "
            "(portal del suscriptor).",
            ["typescript", "react"],
            (24, 29),
            "medium",
        ),
        (
            "Subscription communications",
            "Reminder, charge and shipment emails for subscribers (comunicaciones de suscripción).",
            ["python", "sendgrid"],
            (12, 15),
            "low",
        ),
    ],
    "Legacy POS Stock Sync": [
        (
            "FTP/SFTP connector to legacy POS exports",
            "Nightly retrieval of CSV exports from a legacy point-of-sale over FTP/SFTP "
            "(TPV antiguo, ficheros CSV, FTP).",
            ["python"],
            (12, 15),
            "low",
        ),
        (
            "Stock file parser & ETL",
            "Parsing, validation and normalisation of legacy POS stock files into the shop "
            "inventory (parser de ficheros de stock, ETL).",
            ["python", "postgresql"],
            (22, 27),
            "medium",
        ),
        (
            "Stock reconciliation & oversell prevention",
            "Reconciliation between POS stock and web orders placed since the last export so "
            "nothing is sold twice (conciliación de stock, sobreventa).",
            ["python", "postgresql"],
            (30, 36),
            "high",
        ),
        (
            "Real-time stock reservation",
            "Atomic stock reservation and decrement at checkout with release on timeout "
            "(reserva de stock en tiempo real).",
            ["python", "redis"],
            (20, 24),
            "high",
        ),
    ],
    "Orders, Returns & Invoicing": [
        (
            "Order state machine",
            "Order lifecycle states, transitions and events from payment to delivery "
            "(estados del pedido).",
            ["python", "postgresql"],
            (20, 24),
            "medium",
        ),
        (
            "Returns (RMA) workflow",
            "Customer return requests, approval, return labels and restocking (devoluciones, RMA).",
            ["python", "postgresql"],
            (26, 32),
            "medium",
        ),
        (
            "Refund processing",
            "Full and partial refunds through the payment gateway with audit "
            "(reembolsos, devolución del importe).",
            ["python", "stripe"],
            (16, 20),
            "medium",
        ),
        (
            "Invoice PDF generation with legal numbering",
            "Legally numbered invoices and credit notes rendered to PDF "
            "(factura PDF, numeración legal, abonos).",
            ["python"],
            (20, 25),
            "medium",
        ),
        (
            "Picking & packing workflow",
            "Warehouse pick lists, packing confirmation and internal order notes "
            "(preparación de pedidos, picking).",
            ["python", "typescript"],
            (18, 22),
            "medium",
        ),
    ],
}

# ---------------------------------------------------------------------------
# Domain B — corporate learning, certification and HR integration.
# ---------------------------------------------------------------------------

LEARNING_MODULES: dict[str, list[Template]] = {
    "Learning Content & Access": [
        (
            "Content access control by audience",
            "Course visibility and enrolment rules by department, role and site "
            "(acceso a contenidos por departamento y puesto).",
            ["python", "postgresql"],
            (16, 20),
            "medium",
        ),
        (
            "Learning evidence logging",
            "Immutable log of views, completions and attempts as audit evidence "
            "(evidencias de formación, registro).",
            ["python", "postgresql"],
            (14, 17),
            "medium",
        ),
    ],
    "Assessments": [
        (
            "Assessment configuration",
            "Exam settings: question pools, time limits, passing score and attempt rules "
            "(configuración de exámenes, nota de corte).",
            ["python", "postgresql"],
            (18, 22),
            "medium",
        ),
        (
            "Attempt lifecycle & evidence",
            "Exam attempts with start, autosave, submission and evidence capture "
            "(intentos de examen, evidencias).",
            ["python", "postgresql"],
            (22, 27),
            "medium",
        ),
        (
            "Rubric-based grading by instructors",
            "Practical assignments graded by an instructor with a rubric and feedback "
            "(corrección con rúbrica, formador).",
            ["typescript", "python"],
            (24, 30),
            "medium",
        ),
        (
            "Retake policy & results",
            "Results publication, retake limits and cool-down periods "
            "(repetición de examen, resultados).",
            ["python"],
            (12, 15),
            "low",
        ),
        (
            "Remote proctoring integration",
            "Remote supervised exams with identity check and browser lockdown through a "
            "proctoring vendor, plus review of flagged sessions "
            "(examen supervisado en remoto, control de identidad).",
            ["python", "typescript"],
            (40, 48),
            "high",
        ),
    ],
    "Certificates": [
        (
            "Verifiable certificate codes",
            "Unique verifiable codes / QR on each certificate (código verificable, certificado).",
            ["python"],
            (10, 12),
            "low",
        ),
        (
            "Certificate PDF rendering & storage",
            "Branded certificate PDFs rendered and stored per completion "
            "(certificado PDF, almacenamiento).",
            ["python"],
            (12, 15),
            "low",
        ),
        (
            "Certificate issuance workflow",
            "Automatic issuance on completion with approval for special cases "
            "(emisión de certificados).",
            ["python", "celery"],
            (16, 20),
            "medium",
        ),
        (
            "Public certificate verification portal",
            "Public page to verify a certificate by code (portal de verificación).",
            ["typescript", "react"],
            (14, 17),
            "low",
        ),
        (
            "Certificate revocation & reissue",
            "Revoking, reissuing and versioning certificates with audit (revocación, reemisión).",
            ["python", "postgresql"],
            (14, 17),
            "medium",
        ),
    ],
    "Certification Expiry & Compliance": [
        (
            "Certification expiry reminders",
            "Scheduler that tracks certification expiry dates and sends reminders to the "
            "employee and the manager ahead of time (caducidad de certificaciones, avisos).",
            ["python", "celery"],
            (18, 22),
            "medium",
        ),
        (
            "Mandatory training compliance dashboard",
            "Compliance rates of mandatory training per site and department, with drill-down "
            "(cumplimiento de formación obligatoria por planta).",
            ["python", "typescript"],
            (26, 32),
            "medium",
        ),
        (
            "Audit evidence package export",
            "Export bundle of each employee's training evidence and certificates for "
            "external auditors (informe de auditoría, evidencias).",
            ["python"],
            (20, 24),
            "medium",
        ),
    ],
    "HR System Integration": [
        (
            "HRIS data mapping & identifiers",
            "Mapping employees, departments, positions and managers from the HR system "
            "(SuccessFactors, RR. HH., mapeo de datos).",
            ["python"],
            (14, 17),
            "medium",
        ),
        (
            "HRIS API authentication & connectivity",
            "OAuth / certificate authentication and connectivity to the HR system API "
            "(SuccessFactors, conexión).",
            ["python"],
            (12, 15),
            "medium",
        ),
        (
            "Daily HR sync: transform & upsert",
            "Daily job that transforms and upserts hires, leavers and position changes "
            "(sincronización diaria, altas, bajas, cambios de puesto).",
            ["python", "celery"],
            (24, 30),
            "high",
        ),
        (
            "Automatic training assignment by position",
            "Rules that assign the mandatory courses of a new position when an employee "
            "changes role (asignación automática por puesto).",
            ["python", "postgresql"],
            (24, 28),
            "high",
        ),
        (
            "Sync error handling & idempotency",
            "Retries, dead-letter handling and idempotent upserts for the HR sync "
            "(reintentos, idempotencia).",
            ["python"],
            (12, 15),
            "medium",
        ),
        (
            "HR sync admin console",
            "Admin screen to inspect sync runs, errors and re-run a sync "
            "(consola de sincronización).",
            ["typescript", "react"],
            (14, 17),
            "low",
        ),
        (
            "HR integration sandbox tests",
            "Integration tests against the HR system sandbox tenant "
            "(pruebas de integración, sandbox).",
            ["python"],
            (16, 20),
            "medium",
        ),
    ],
    "Learning Admin": [
        (
            "Role-to-training mapping UI",
            "Admin UI to define which courses each position requires (matriz puesto-formación).",
            ["typescript", "react"],
            (20, 24),
            "medium",
        ),
        (
            "Bulk import of training history",
            "Excel import of historical training records with validation and error report "
            "(importación de históricos desde Excel).",
            ["python"],
            (18, 22),
            "medium",
        ),
    ],
}

# ---------------------------------------------------------------------------
# Shared platform work both domains need, in their wording.
# ---------------------------------------------------------------------------

PLATFORM_MODULES: dict[str, list[Template]] = {
    "Platform & Operations": [
        (
            "Backups & disaster recovery",
            "Automated backups, restore drills and a documented recovery plan "
            "(copias de seguridad, recuperación ante desastres).",
            ["terraform", "postgresql"],
            (16, 20),
            "medium",
        ),
        (
            "CDN, caching & performance tuning",
            "CDN in front of the storefront or app, HTTP caching and query tuning "
            "(rendimiento, caché).",
            ["terraform", "redis"],
            (18, 22),
            "medium",
        ),
        (
            "Admin shell & internal roles",
            "Back-office shell with internal roles and permissions "
            "(panel de administración, roles internos).",
            ["typescript", "react"],
            (20, 24),
            "medium",
        ),
        (
            "Admin audit trail",
            "Who-changed-what audit trail for back-office actions (registro de auditoría).",
            ["python", "postgresql"],
            (12, 15),
            "medium",
        ),
        (
            "Social login & account linking",
            "Google sign-in with account linking to existing email accounts "
            "(login con Google, vinculación de cuentas).",
            ["python"],
            (12, 15),
            "medium",
        ),
        (
            "Customer account dashboard",
            "Customer area with profile, addresses and order history (área de cliente).",
            ["typescript", "react"],
            (18, 22),
            "medium",
        ),
    ],
    "Privacy & GDPR": [
        (
            "Cookie consent management",
            "Consent banner and preference storage compliant with GDPR "
            "(gestión de cookies y consentimiento, RGPD).",
            ["typescript"],
            (8, 10),
            "low",
        ),
        (
            "Data subject rights: export & deletion",
            "Self-service export and deletion of personal data with audit "
            "(derechos RGPD, exportación y borrado).",
            ["python", "postgresql"],
            (18, 22),
            "medium",
        ),
        (
            "Notification preferences & opt-outs",
            "Per-channel notification preferences and legal opt-outs (preferencias de avisos).",
            ["python"],
            (10, 12),
            "low",
        ),
    ],
}

# The hours profile of each template; anything not listed is ``tight``.
PROFILE: dict[str, str] = {
    # commerce — Jev's band
    "Search synonyms, typos & no-results analytics": "noisy",
    "VAT calculation & invoice data capture": "noisy",
    "Delivery options & delivery-day selector": "noisy",
    "Subscription lifecycle: pause, skip & cancel": "noisy",
    "Batch order generation for subscriptions": "noisy",
    "Stock file parser & ETL": "noisy",
    "Real-time stock reservation": "noisy",
    "Returns (RMA) workflow": "noisy",
    "Invoice PDF generation with legal numbering": "noisy",
    "Food & wine pairing recommendations": "contested",
    "Box curation & cycle management": "contested",
    "Stock reconciliation & oversell prevention": "contested",
    "Picking & packing workflow": "contested",
    # learning — Jev's band
    "Attempt lifecycle & evidence": "noisy",
    "Certificate issuance workflow": "noisy",
    "Certificate revocation & reissue": "noisy",
    "Certification expiry reminders": "noisy",
    "Mandatory training compliance dashboard": "noisy",
    "HRIS data mapping & identifiers": "noisy",
    "Daily HR sync: transform & upsert": "noisy",
    "HR integration sandbox tests": "noisy",
    "Role-to-training mapping UI": "noisy",
    "Remote proctoring integration": "contested",
    "Automatic training assignment by position": "contested",
    "Audit evidence package export": "contested",
    # shared
    "CDN, caching & performance tuning": "noisy",
    "Data subject rights: export & deletion": "noisy",
}

ABBREV = {
    "Product Catalog & Pricing": "CAT",
    "Search & Recommendations": "SRCH",
    "Checkout & Taxes": "CHK",
    "Subscriptions": "SUB",
    "Legacy POS Stock Sync": "POS",
    "Orders, Returns & Invoicing": "ORD",
    "Learning Content & Access": "LCA",
    "Assessments": "ASMT",
    "Certificates": "CERT",
    "Certification Expiry & Compliance": "EXP",
    "HR System Integration": "HRIS",
    "Learning Admin": "LADM",
    "Platform & Operations": "PLAT",
    "Privacy & GDPR": "GDPR",
}

# Per domain: sector (an existing ``Sector`` literal — a new value would 422 on
# ingest), the modules it draws from, and project headers. The header is a large part
# of a short chunk's text, so it places the domain without merging the tasks.
DOMAINS = {
    "commerce": {
        "sector": "ecommerce",
        "modules": {**COMMERCE_MODULES, **PLATFORM_MODULES},
        "summaries": [
            "Gourmet food online shop: catalog with weight-based pricing, monthly "
            "subscription box, stock sync with a legacy point-of-sale and returns",
            "Specialty retail e-commerce replatforming: checkout with VAT and delivery "
            "days, order management, invoicing and legacy POS inventory sync",
            "Subscription commerce platform: recurring boxes, subscriber portal, batch "
            "fulfilment orders and payment retries",
        ],
        "clients": ["DespensaSur", "Ultramarinos Rey", "VinoyQueso", "MercaFino", "LaAlacena"],
        "tech": ["python_django", "node_nestjs", "ruby_on_rails"],
    },
    "learning": {
        "sector": "education",
        "modules": {**LEARNING_MODULES, **PLATFORM_MODULES},
        "summaries": [
            "Corporate academy for an industrial group: mandatory safety training, exams, "
            "verifiable certificates and certification expiry tracking",
            "Compliance training platform: HR system integration, automatic course "
            "assignment by position and audit evidence for regulators",
            "Enterprise learning platform: assessments with remote proctoring, rubric "
            "grading, certificates and compliance dashboards per site",
        ],
        "clients": [
            "EnergiaNova",
            "Grupo Vértice",
            "AceroNorte",
            "RedEléctrica Sur",
            "Aguas Lumen",
        ],
        "tech": ["python_fastapi", "java_spring", "node_nestjs"],
    },
}
YEARS = [2022, 2023, 2024, 2025]
COUNTRIES = ["ES", "ES", "ES", "PT"]
SUFFIXES = ["S.L.", "S.A.", "Group"]


def draw_hours(rng: random.Random, name: str, lo: int, hi: int, nth: int) -> int:
    """Hours of the ``nth`` occurrence (0-based) of a template, shaped by its profile."""
    base = rng.randint(lo, hi)
    profile = PROFILE.get(name, "tight")
    if profile == "noisy" and nth % NOISY_OUTLIER_EVERY == 1:
        return round(base * rng.uniform(*NOISY_FACTOR))
    if profile == "contested" and nth % CONTESTED_LARGE_EVERY == 1:
        return round(base * rng.uniform(*CONTESTED_FACTOR))
    return base


def _build_project(rng: random.Random, index: int, seen: dict[str, int]) -> dict:
    """Synthesise one historical project as a Budget dict (tasks = components)."""
    domain_name = "commerce" if index % 2 else "learning"
    domain = DOMAINS[domain_name]
    catalog: dict[str, list[Template]] = domain["modules"]
    year = rng.choice(YEARS)

    modules = list(catalog)
    chosen = rng.sample(modules, k=max(3, len(modules) - rng.randint(0, 2)))

    components: list[dict] = []
    counters: dict[str, int] = {}
    for module in chosen:
        templates = catalog[module]
        k = min(len(templates), rng.randint(max(2, len(templates) - 2), len(templates)))
        for name, desc, tech, (lo, hi), complexity in rng.sample(templates, k=k):
            counters[module] = counters.get(module, 0) + 1
            nth = seen.get(name, 0)
            seen[name] = nth + 1
            components.append(
                {
                    "component_id": f"{ABBREV[module]}-{counters[module]:03d}",
                    "name": name,
                    "description": desc,
                    "module": module,
                    "tech_stack": rng.sample(tech, k=min(len(tech), rng.randint(1, len(tech)))),
                    "estimated_hours": draw_hours(rng, name, lo, hi, nth),
                    "complexity": complexity,
                    "dependencies": [],
                }
            )

    return {
        "budget_id": f"DEMO-{year}-{index:04d}",
        "client_metadata": {
            "name": f"{rng.choice(domain['clients'])} {rng.choice(SUFFIXES)}",
            "sector": domain["sector"],
            "country": rng.choice(COUNTRIES),
        },
        "project_summary": rng.choice(domain["summaries"]),
        "main_technology": rng.choice(domain["tech"]),
        "year": year,
        "total_estimated_hours": sum(c["estimated_hours"] for c in components),
        "components": components,
    }


def generate_corpus(count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED) -> list[dict]:
    """Generate ``count`` demo projects (alternating domains), deterministically from ``seed``."""
    rng = random.Random(seed)
    seen: dict[str, int] = {}  # occurrences per template so far
    return [_build_project(rng, i + 1, seen) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and ingest the demo task corpus.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of projects.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (reproducible).")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="Output JSON path.")
    parser.add_argument("--base-url", default=None, help="AI service base URL (else auto-probe).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true", help="Write JSON, do not ingest.")
    mode.add_argument("--ingest-only", action="store_true", help="Ingest an existing JSON file.")
    parser.add_argument("--ingest", action="store_true", help="Generate AND ingest.")
    args = parser.parse_args()

    if args.ingest_only:
        corpus = json.loads(args.out.read_text())
        print(f"Loaded {len(corpus)} projects from {args.out}.")
    else:
        corpus = generate_corpus(count=args.count, seed=args.seed)
        args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False))
        tasks = sum(len(p["components"]) for p in corpus)
        print(f"Wrote {len(corpus)} projects / {tasks} tasks → {args.out}")

    if args.generate_only:
        return
    if args.ingest or args.ingest_only:
        ingest_corpus(
            corpus,
            base_url=args.base_url,
            document_type=DOCUMENT_TYPE,
            chunk_type=CHUNK_TYPE,
            source_prefix=SOURCE_PREFIX,
        )
    else:
        print("(generation only; pass --ingest to load into pgvector)")


if __name__ == "__main__":
    main()
