"""Base schema types for LLM-native research artifacts.

These are the foundational building blocks that all artifact types share.
First-class citizens: causal relationships, conditions, uncertainty, provenance.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Provenance — where did this data come from?
# ---------------------------------------------------------------------------

class ProvenanceType(str, Enum):
    PAPER = "paper"
    DATASET = "dataset"
    CODE = "code"
    HUMAN_ANNOTATION = "human_annotation"
    LLM_EXTRACTION = "llm_extraction"
    LLM_SYNTHESIS = "llm_synthesis"
    EXPERIMENT = "experiment"
    DERIVED = "derived"


class Provenance(BaseModel):
    """Tracks the origin and transformation history of data."""

    source_type: ProvenanceType
    source_id: str = Field(
        description="DOI, URL, dataset ID, or other unique identifier"
    )
    source_title: str | None = None
    extraction_method: str | None = Field(
        default=None,
        description="How data was extracted (e.g., 'claude-3.5-sonnet table extraction')",
    )
    extraction_date: datetime = Field(default_factory=_utcnow)
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the extraction accuracy",
    )
    parent_artifact_ids: list[str] = Field(
        default_factory=list,
        description="IDs of artifacts this was derived from",
    )
    notes: str | None = None


# ---------------------------------------------------------------------------
# Uncertainty — explicit representation of what we don't know
# ---------------------------------------------------------------------------

class UncertaintyType(str, Enum):
    STATISTICAL = "statistical"
    SYSTEMATIC = "systematic"
    MEASUREMENT = "measurement"
    MODEL = "model"
    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"


class UncertaintyEstimate(BaseModel):
    """Explicit, machine-readable uncertainty representation."""

    uncertainty_type: UncertaintyType
    value: float | None = Field(
        default=None, description="Numeric uncertainty (e.g., std dev)"
    )
    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence_level: float | None = Field(
        default=None,
        description="Confidence level for intervals (e.g., 0.95)",
    )
    description: str | None = Field(
        default=None,
        description="Natural language description of the uncertainty",
    )


# ---------------------------------------------------------------------------
# Conditions — under what circumstances does a claim hold?
# ---------------------------------------------------------------------------

class ConditionType(str, Enum):
    PRECONDITION = "precondition"
    ASSUMPTION = "assumption"
    CONSTRAINT = "constraint"
    LIMITATION = "limitation"
    SCOPE = "scope"
    ENVIRONMENT = "environment"


class Condition(BaseModel):
    """A condition under which a claim or result holds."""

    condition_type: ConditionType
    description: str
    formal_expression: str | None = Field(
        default=None,
        description="Machine-parseable expression (e.g., 'dataset_size > 1000')",
    )
    is_verified: bool | None = Field(
        default=None,
        description="Whether this condition was verified in the source",
    )


# ---------------------------------------------------------------------------
# Causal Relationships
# ---------------------------------------------------------------------------

class CausalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    SUGGESTED = "suggested"
    CORRELATIONAL = "correlational"


class CausalRelationship(BaseModel):
    """Explicit representation of causal claims."""

    cause: str = Field(description="The causal factor")
    effect: str = Field(description="The effect or outcome")
    strength: CausalStrength
    mechanism: str | None = Field(
        default=None, description="Proposed mechanism"
    )
    conditions: list[Condition] = Field(default_factory=list)
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of evidence supporting this relationship",
    )
    confounders: list[str] = Field(
        default_factory=list,
        description="Known or suspected confounding factors",
    )


# ---------------------------------------------------------------------------
# Evidence and Claims — the core epistemic units
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    EXPERIMENTAL = "experimental"
    OBSERVATIONAL = "observational"
    THEORETICAL = "theoretical"
    SIMULATION = "simulation"
    META_ANALYSIS = "meta_analysis"


class Evidence(BaseModel):
    """A piece of evidence supporting or contradicting a claim."""

    id: str = Field(default_factory=_new_id)
    evidence_type: EvidenceType
    description: str
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data (metrics, measurements, etc.)",
    )
    uncertainty: UncertaintyEstimate | None = None
    conditions: list[Condition] = Field(default_factory=list)
    provenance: Provenance | None = None
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the evidence (0=very weak, 1=very strong)",
    )


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTESTED = "contested"
    REFUTED = "refuted"
    PRELIMINARY = "preliminary"
    ESTABLISHED = "established"


class Claim(BaseModel):
    """A scientific claim with explicit evidence, conditions, and status."""

    id: str = Field(default_factory=_new_id)
    statement: str = Field(description="The claim in natural language")
    formal_statement: str | None = Field(
        default=None, description="Machine-parseable version"
    )
    status: ClaimStatus = ClaimStatus.PRELIMINARY
    evidence_for: list[Evidence] = Field(default_factory=list)
    evidence_against: list[Evidence] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    causal_relationships: list[CausalRelationship] = Field(default_factory=list)
    related_claim_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the claim",
    )


# ---------------------------------------------------------------------------
# Artifact Metadata — shared across all artifact types
# ---------------------------------------------------------------------------

class ArtifactType(str, Enum):
    EXPERIMENT_RESULT = "experiment_result"
    METHOD_COMPARISON = "method_comparison"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS = "hypothesis"
    DATASET_DESCRIPTION = "dataset_description"


class ArtifactMetadata(BaseModel):
    """Metadata common to all LLM-native research artifacts."""

    id: str = Field(default_factory=_new_id)
    artifact_type: ArtifactType
    title: str
    description: str
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    provenance: list[Provenance] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    domain: str = Field(
        default="machine_learning",
        description="Research domain (e.g., 'machine_learning', 'biology')",
    )
    schema_version: str = "0.1.0"
