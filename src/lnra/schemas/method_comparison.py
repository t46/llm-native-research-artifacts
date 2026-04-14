"""Method Comparison Artifact schema.

Represents a structured comparison between multiple methods/approaches,
making it trivial for AI agents to answer questions like "Which method
is best for X under conditions Y?" without reading full papers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .base import (
    ArtifactMetadata,
    ArtifactType,
    Claim,
    CausalRelationship,
    Condition,
    Evidence,
    Provenance,
    UncertaintyEstimate,
    _new_id,
)


class MethodCategory(str, Enum):
    PROPOSED = "proposed"
    BASELINE = "baseline"
    STATE_OF_THE_ART = "state_of_the_art"
    ABLATION = "ablation"
    ORACLE = "oracle"


class MethodDescription(BaseModel):
    """Structured description of a method."""

    id: str = Field(default_factory=_new_id)
    name: str
    category: MethodCategory = MethodCategory.BASELINE
    description: str = ""
    key_innovation: str | None = Field(
        default=None,
        description="What makes this method different from others",
    )
    preconditions: list[Condition] = Field(
        default_factory=list,
        description="What must be true for this method to work",
    )
    limitations: list[str] = Field(default_factory=list)
    computational_cost: str | None = Field(
        default=None,
        description="Relative or absolute compute cost",
    )
    paper_reference: str | None = None
    year: int | None = None
    components: list[str] = Field(
        default_factory=list,
        description="Key components/modules of the method",
    )


class ComparisonDimension(BaseModel):
    """A dimension along which methods are compared."""

    name: str = Field(description="Dimension name (e.g., 'accuracy', 'speed', 'memory')")
    description: str | None = None
    unit: str | None = None
    higher_is_better: bool = True
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative importance of this dimension",
    )


class MethodScore(BaseModel):
    """A method's score on a specific comparison dimension."""

    method_id: str
    dimension_name: str
    value: float
    uncertainty: UncertaintyEstimate | None = None
    conditions: list[Condition] = Field(default_factory=list)
    notes: str | None = None


class ComparisonResult(BaseModel):
    """Result of comparing methods along a dimension."""

    dimension: ComparisonDimension
    scores: list[MethodScore] = Field(default_factory=list)
    winner_id: str | None = Field(
        default=None,
        description="ID of the winning method (if clear)",
    )
    statistical_significance: float | None = Field(
        default=None,
        description="p-value if statistical test was performed",
    )
    notes: str | None = None

    def get_ranking(self) -> list[MethodScore]:
        """Get scores ranked from best to worst."""
        return sorted(
            self.scores,
            key=lambda s: s.value,
            reverse=self.dimension.higher_is_better,
        )


class TradeoffAnalysis(BaseModel):
    """Analysis of tradeoffs between methods."""

    description: str
    methods_involved: list[str] = Field(
        description="IDs of methods involved in the tradeoff"
    )
    dimensions_involved: list[str] = Field(
        description="Names of dimensions involved in the tradeoff"
    )
    recommendation: str | None = None
    conditions: list[Condition] = Field(
        default_factory=list,
        description="Under what conditions this tradeoff analysis holds",
    )


class MethodComparisonArtifact(BaseModel):
    """A complete method comparison artifact.

    Enables AI agents to quickly answer: "Which method is best for X?"
    "What are the tradeoffs?" "Under what conditions does method A beat B?"
    """

    metadata: ArtifactMetadata = Field(
        default_factory=lambda: ArtifactMetadata(
            artifact_type=ArtifactType.METHOD_COMPARISON,
            title="",
            description="",
        )
    )
    methods: list[MethodDescription] = Field(default_factory=list)
    dimensions: list[ComparisonDimension] = Field(default_factory=list)
    results: list[ComparisonResult] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    causal_relationships: list[CausalRelationship] = Field(default_factory=list)
    tradeoffs: list[TradeoffAnalysis] = Field(default_factory=list)
    recommendation: str | None = Field(
        default=None,
        description="Overall recommendation from the comparison",
    )
    domain_context: str | None = Field(
        default=None,
        description="Domain-specific context for the comparison",
    )

    def get_method_by_name(self, name: str) -> MethodDescription | None:
        """Find a method by name (case-insensitive)."""
        name_lower = name.lower()
        for m in self.methods:
            if m.name.lower() == name_lower:
                return m
        return None

    def get_best_method(self, dimension_name: str) -> MethodDescription | None:
        """Get the best method for a given dimension."""
        for result in self.results:
            if result.dimension.name == dimension_name and result.winner_id:
                for m in self.methods:
                    if m.id == result.winner_id:
                        return m
        return None

    def get_method_profile(self, method_id: str) -> dict[str, Any]:
        """Get a complete profile of a method across all dimensions."""
        method = next((m for m in self.methods if m.id == method_id), None)
        if not method:
            return {}
        profile: dict[str, Any] = {
            "name": method.name,
            "category": method.category.value,
            "scores": {},
            "rank": {},
        }
        for result in self.results:
            for score in result.scores:
                if score.method_id == method_id:
                    profile["scores"][result.dimension.name] = score.value
                    ranking = result.get_ranking()
                    for i, s in enumerate(ranking):
                        if s.method_id == method_id:
                            profile["rank"][result.dimension.name] = i + 1
                            break
        return profile

    def get_preconditions_for(self, method_name: str) -> list[str]:
        """Get all preconditions for a given method."""
        method = self.get_method_by_name(method_name)
        if not method:
            return []
        return [c.description for c in method.preconditions]
