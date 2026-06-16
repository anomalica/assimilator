"""Algorithmic evidence scoring for claims.

Scores are computed from graph properties, not assigned by humans.
The methodology is transparent and reproducible.

Scoring factors:
- Number of independent records corroborating a claim
- Attestation depth (first-hand > second-hand > third-hand)
- Claim type weight (measurement > testimony > observation > hearsay > opinion)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from assimilator.database import get_corroborations, get_independent_source_count
from anomalica_common.digest.models import AttestationLevel, ClaimType


ATTESTATION_WEIGHTS = {
    AttestationLevel.first_hand: 1.0,
    AttestationLevel.second_hand: 0.6,
    AttestationLevel.third_hand: 0.3,
}

CLAIM_TYPE_WEIGHTS = {
    ClaimType.measurement: 0.9,
    ClaimType.testimony: 0.8,
    ClaimType.observation: 0.75,
    ClaimType.administrative: 0.7,
    ClaimType.hearsay: 0.4,
    ClaimType.opinion: 0.3,
}


@dataclass
class ScoreBreakdown:
    score: float
    record_count: int
    corroboration_count: int
    attestation: str
    claim_type: str
    base_weight: float
    corroboration_bonus: float
    components: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"{self.record_count} record(s)",
            self.attestation,
            self.claim_type,
        ]
        if self.corroboration_count > 0:
            parts.append(f"{self.corroboration_count} corroboration(s)")
        parts.append(f"score: {self.score:.2f}")
        return ", ".join(parts)


def score_claim(conn: sqlite3.Connection, claim_id: str) -> ScoreBreakdown:
    """Calculate an evidence score for a single claim."""
    row = conn.execute(
        "SELECT claim_type, attestation, record_id FROM claims WHERE id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        return ScoreBreakdown(
            score=0.0,
            record_count=0,
            corroboration_count=0,
            attestation="unknown",
            claim_type="unknown",
            base_weight=0.0,
            corroboration_bonus=0.0,
        )

    claim_type = ClaimType(row[0])
    attestation = AttestationLevel(row[1]) if row[1] else None

    # Base weight from claim type and attestation (absent attestation = neutral)
    type_weight = CLAIM_TYPE_WEIGHTS.get(claim_type, 0.5)
    attestation_weight = ATTESTATION_WEIGHTS.get(attestation, 0.5)
    base_weight = type_weight * attestation_weight

    # Corroboration from independent sources (not just records)
    # Two claims from the same speaker in different records share a provenance
    # root and count as one source, not two.
    corroborations = get_corroborations(conn, claim_id)
    source_count = get_independent_source_count(conn, claim_id)

    # Noisy-OR: each independent corroborating record increases confidence
    # The intuition: if one source is wrong with probability (1 - base_weight),
    # two independent sources are both wrong with probability (1 - base_weight)^2
    if source_count <= 1:
        combined = base_weight
    else:
        product = 1.0
        for _ in range(source_count):
            product *= 1.0 - base_weight
        combined = 1.0 - product

    final = min(1.0, combined)

    return ScoreBreakdown(
        score=final,
        record_count=source_count,
        corroboration_count=len(corroborations),
        attestation=attestation.value if attestation else "unattributed",
        claim_type=claim_type.value,
        base_weight=base_weight,
        corroboration_bonus=combined - base_weight,
        components={
            "type_weight": type_weight,
            "attestation_weight": attestation_weight,
            "base": base_weight,
            "combined": combined,
        },
    )


def score_all_claims(conn: sqlite3.Connection) -> dict[str, ScoreBreakdown]:
    """Score every claim in the database."""
    rows = conn.execute("SELECT id FROM claims").fetchall()
    return {row[0]: score_claim(conn, row[0]) for row in rows}


def tier_label(score: float) -> str:
    """Convert a numeric score to a human-readable tier label.

    Labels describe evidence strength, not editorial judgement about the claim.
    """
    if score >= 0.85:
        return "strong"
    if score >= 0.65:
        return "moderate"
    if score >= 0.40:
        return "weak"
    return "insufficient"
