"""Experiment Result Artifact schema.

Represents the structured output of a scientific experiment, including
setup, metrics, results, and interpretations — all machine-readable.
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


class HyperparameterType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class Hyperparameter(BaseModel):
    """A hyperparameter used in the experiment."""

    name: str
    value: Any
    param_type: HyperparameterType = HyperparameterType.CONTINUOUS
    search_range: dict[str, Any] | None = Field(
        default=None,
        description="Range explored during tuning (e.g., {'min': 1e-5, 'max': 1e-1})",
    )
    is_tuned: bool = False


class DatasetInfo(BaseModel):
    """Information about a dataset used in the experiment."""

    name: str
    version: str | None = None
    size: int | str | None = Field(default=None, description="Number of samples or description of size")
    splits: dict[str, int | str] = Field(
        default_factory=dict,
        description="Split name -> number of samples",
    )
    preprocessing: list[str] = Field(
        default_factory=list,
        description="Preprocessing steps applied",
    )
    url: str | None = None
    characteristics: dict[str, Any] = Field(
        default_factory=dict,
        description="Key dataset characteristics (e.g., num_classes, imbalance_ratio)",
    )


class ExperimentalSetup(BaseModel):
    """Full description of the experimental setup."""

    task: str = Field(description="What task is being evaluated")
    methodology: str = Field(description="How the experiment was conducted")
    datasets: list[DatasetInfo] = Field(default_factory=list)
    hyperparameters: list[Hyperparameter] = Field(default_factory=list)
    hardware: str | None = Field(
        default=None, description="Compute environment (e.g., '8x A100')"
    )
    software_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Key software versions (e.g., {'pytorch': '2.1', 'cuda': '12.0'})",
    )
    random_seeds: list[int] = Field(default_factory=list)
    num_runs: int = Field(default=1, description="Number of independent runs")
    evaluation_protocol: str | None = Field(
        default=None,
        description="How results are evaluated (e.g., '5-fold cross-validation')",
    )
    baselines: list[str] = Field(
        default_factory=list,
        description="Baseline methods compared against",
    )
    conditions: list[Condition] = Field(
        default_factory=list,
        description="Experimental conditions and assumptions",
    )


class Metric(BaseModel):
    """A measured metric with full uncertainty information."""

    name: str = Field(description="Metric name (e.g., 'accuracy', 'F1')")
    value: float
    uncertainty: UncertaintyEstimate | None = None
    higher_is_better: bool = True
    unit: str | None = None
    dataset: str | None = Field(
        default=None, description="Which dataset this metric is from"
    )
    split: str | None = Field(
        default=None, description="Which split (e.g., 'test', 'val')"
    )
    conditions: list[Condition] = Field(
        default_factory=list,
        description="Conditions under which this metric holds",
    )


class Result(BaseModel):
    """A result for a specific method/configuration."""

    id: str = Field(default_factory=_new_id)
    method_name: str
    metrics: list[Metric] = Field(default_factory=list)
    is_proposed: bool = Field(
        default=False,
        description="True if this is the proposed method (vs. baseline)",
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict,
        description="Specific configuration for this run",
    )


class ExperimentResultArtifact(BaseModel):
    """A complete experiment result artifact.

    This is the primary artifact type for representing ML experiment outcomes
    in a machine-readable format.
    """

    metadata: ArtifactMetadata = Field(
        default_factory=lambda: ArtifactMetadata(
            artifact_type=ArtifactType.EXPERIMENT_RESULT,
            title="",
            description="",
        )
    )
    setup: ExperimentalSetup
    results: list[Result] = Field(default_factory=list)
    claims: list[Claim] = Field(
        default_factory=list,
        description="Claims made based on the results",
    )
    causal_relationships: list[CausalRelationship] = Field(
        default_factory=list,
        description="Causal claims made in the paper",
    )
    ablation_results: list[Result] = Field(
        default_factory=list,
        description="Ablation study results",
    )
    failure_cases: list[str] = Field(
        default_factory=list,
        description="Known failure cases or limitations",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Key findings in natural language",
    )
    raw_tables: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw tabular data extracted from the paper",
    )

    def get_best_result(self, metric_name: str) -> Result | None:
        """Get the result with the best value for a given metric."""
        best: Result | None = None
        best_val: float | None = None
        for result in self.results:
            for m in result.metrics:
                if m.name == metric_name:
                    if best_val is None:
                        best = result
                        best_val = m.value
                    elif m.higher_is_better and m.value > best_val:
                        best = result
                        best_val = m.value
                    elif not m.higher_is_better and m.value < best_val:
                        best = result
                        best_val = m.value
        return best

    def get_metric_comparison(self, metric_name: str) -> list[dict]:
        """Get a comparison of all methods for a given metric."""
        comparison = []
        for result in self.results:
            for m in result.metrics:
                if m.name == metric_name:
                    entry = {
                        "method": result.method_name,
                        "value": m.value,
                        "is_proposed": result.is_proposed,
                    }
                    if m.uncertainty:
                        entry["uncertainty"] = m.uncertainty.model_dump()
                    comparison.append(entry)
        return sorted(comparison, key=lambda x: x["value"], reverse=True)

    def get_claims_with_evidence(self) -> list[dict]:
        """Get all claims with their supporting evidence."""
        return [
            {
                "claim": claim.statement,
                "status": claim.status.value,
                "confidence": claim.confidence,
                "evidence_for": [e.description for e in claim.evidence_for],
                "evidence_against": [e.description for e in claim.evidence_against],
                "conditions": [c.description for c in claim.conditions],
            }
            for claim in self.claims
        ]
