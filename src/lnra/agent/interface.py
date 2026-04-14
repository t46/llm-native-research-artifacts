"""Agent interface for LLM-native research artifacts.

Provides three core operations:
- query(): Ask questions about artifacts (preconditions, confidence intervals, etc.)
- compose(): Integrate multiple artifacts to generate new insights
- diff(): Detect differences and contradictions between artifacts
"""

from __future__ import annotations

import json
import logging
from typing import Any, Union

import anthropic

from ..schemas.base import Claim, CausalRelationship, Condition
from ..schemas.experiment import ExperimentResultArtifact
from ..schemas.method_comparison import MethodComparisonArtifact

logger = logging.getLogger(__name__)

Artifact = Union[ExperimentResultArtifact, MethodComparisonArtifact]


class ArtifactAgent:
    """Agent that can query, compose, and diff LLM-native research artifacts.

    This is the core innovation: instead of reading papers, agents operate
    on structured artifacts programmatically.

    Usage:
        agent = ArtifactAgent()

        # Query
        answer = agent.query(artifact, "What are the preconditions?")

        # Compose
        insights = agent.compose([artifact1, artifact2], "What patterns emerge?")

        # Diff
        differences = agent.diff(artifact1, artifact2)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def _call_claude(self, system: str, user: str) -> str:
        """Make a call to Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from response."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()
        return json.loads(text)

    def _artifact_to_context(self, artifact: Artifact) -> str:
        """Convert artifact to a context string for Claude."""
        data = artifact.model_dump(mode="json")
        return json.dumps(data, indent=2, default=str)

    # ------------------------------------------------------------------
    # query() — Ask questions about a single artifact
    # ------------------------------------------------------------------

    def query(self, artifact: Artifact, question: str) -> dict[str, Any]:
        """Ask a question about an artifact and get a structured answer.

        This is far more precise than asking about a paper, because the
        artifact has explicit structure for conditions, uncertainty, claims, etc.

        Args:
            artifact: The artifact to query
            question: Natural language question

        Returns:
            Structured answer with 'answer', 'evidence', 'confidence', 'caveats'
        """
        # First, try to answer programmatically for common question types
        programmatic = self._try_programmatic_query(artifact, question)
        if programmatic is not None:
            return programmatic

        # Fall back to LLM-augmented query
        context = self._artifact_to_context(artifact)
        system = """You are a research artifact query engine. You have access to a structured
research artifact (JSON) and must answer questions about it precisely.

IMPORTANT: Your answers must be grounded in the artifact data. Do not speculate.

Respond with a JSON object:
{
  "answer": "Direct answer to the question",
  "evidence": ["List of specific data points from the artifact supporting the answer"],
  "confidence": 0.9,  // How confident you are in the answer (0-1)
  "caveats": ["Any caveats or limitations of the answer"],
  "relevant_conditions": ["Any conditions that affect the answer"]
}

Output ONLY valid JSON."""

        response = self._call_claude(
            system=system,
            user=f"Artifact:\n{context}\n\nQuestion: {question}",
        )
        return self._extract_json(response)

    def _try_programmatic_query(
        self, artifact: Artifact, question: str
    ) -> dict[str, Any] | None:
        """Try to answer common queries programmatically without LLM call."""
        q = question.lower()

        if isinstance(artifact, ExperimentResultArtifact):
            return self._programmatic_experiment_query(artifact, q)
        elif isinstance(artifact, MethodComparisonArtifact):
            return self._programmatic_comparison_query(artifact, q)
        return None

    def _programmatic_experiment_query(
        self, artifact: ExperimentResultArtifact, q: str
    ) -> dict[str, Any] | None:
        """Programmatic queries for experiment artifacts."""
        # Best result queries
        if "best" in q and ("result" in q or "method" in q or "performance" in q):
            # Find the first metric mentioned
            for result in artifact.results:
                for metric in result.metrics:
                    if metric.name.lower() in q:
                        best = artifact.get_best_result(metric.name)
                        if best:
                            return {
                                "answer": f"Best method for {metric.name}: {best.method_name}",
                                "evidence": [
                                    f"{m.name}={m.value}"
                                    for m in best.metrics
                                    if m.name == metric.name
                                ],
                                "confidence": 1.0,
                                "caveats": [],
                                "relevant_conditions": [
                                    c.description for m in best.metrics
                                    for c in m.conditions
                                ],
                                "source": "programmatic",
                            }

        # Claims query
        if "claim" in q:
            claims_data = artifact.get_claims_with_evidence()
            if claims_data:
                return {
                    "answer": f"Found {len(claims_data)} claims",
                    "evidence": claims_data,
                    "confidence": 1.0,
                    "caveats": [],
                    "relevant_conditions": [],
                    "source": "programmatic",
                }

        # Key findings
        if "finding" in q or "key" in q:
            if artifact.key_findings:
                return {
                    "answer": f"{len(artifact.key_findings)} key findings",
                    "evidence": artifact.key_findings,
                    "confidence": 1.0,
                    "caveats": [],
                    "relevant_conditions": [],
                    "source": "programmatic",
                }

        return None

    def _programmatic_comparison_query(
        self, artifact: MethodComparisonArtifact, q: str
    ) -> dict[str, Any] | None:
        """Programmatic queries for method comparison artifacts."""
        # Preconditions query
        if "precondition" in q:
            all_preconditions = {}
            for method in artifact.methods:
                if method.preconditions:
                    all_preconditions[method.name] = [
                        c.description for c in method.preconditions
                    ]
            if all_preconditions:
                return {
                    "answer": f"Preconditions for {len(all_preconditions)} methods",
                    "evidence": all_preconditions,
                    "confidence": 1.0,
                    "caveats": [],
                    "relevant_conditions": [],
                    "source": "programmatic",
                }

        # Best method query
        if "best" in q:
            for dim in artifact.dimensions:
                if dim.name.lower() in q:
                    best = artifact.get_best_method(dim.name)
                    if best:
                        return {
                            "answer": f"Best method for {dim.name}: {best.name}",
                            "evidence": [best.description],
                            "confidence": 1.0,
                            "caveats": [
                                f"Limitations: {', '.join(best.limitations)}"
                            ] if best.limitations else [],
                            "relevant_conditions": [
                                c.description for c in best.preconditions
                            ],
                            "source": "programmatic",
                        }

        # Tradeoff query
        if "tradeoff" in q or "trade-off" in q:
            if artifact.tradeoffs:
                return {
                    "answer": f"Found {len(artifact.tradeoffs)} tradeoffs",
                    "evidence": [
                        {
                            "description": t.description,
                            "recommendation": t.recommendation,
                        }
                        for t in artifact.tradeoffs
                    ],
                    "confidence": 1.0,
                    "caveats": [],
                    "relevant_conditions": [],
                    "source": "programmatic",
                }

        return None

    # ------------------------------------------------------------------
    # compose() — Integrate multiple artifacts
    # ------------------------------------------------------------------

    def compose(
        self,
        artifacts: list[Artifact],
        question: str | None = None,
    ) -> dict[str, Any]:
        """Compose multiple artifacts to generate new insights.

        This is the power of structured artifacts: cross-paper analysis
        becomes a programmatic operation rather than reading multiple papers.

        Args:
            artifacts: List of artifacts to compose
            question: Optional guiding question for the composition

        Returns:
            Structured composition with insights, contradictions, and synthesis
        """
        # First, do programmatic cross-artifact analysis
        programmatic_analysis = self._programmatic_compose(artifacts)

        # Then use LLM for deeper synthesis
        contexts = []
        for i, art in enumerate(artifacts):
            contexts.append(f"=== Artifact {i+1} ===\n{self._artifact_to_context(art)}")

        all_context = "\n\n".join(contexts)

        system = """You are a research synthesis engine. You have access to multiple structured
research artifacts and must synthesize insights across them.

You have also been provided with a programmatic cross-artifact analysis.

Respond with a JSON object:
{
  "synthesis": "Overall synthesis of the artifacts",
  "shared_findings": ["Findings that appear across multiple artifacts"],
  "contradictions": [{
    "description": "What contradicts",
    "artifact_indices": [0, 1],
    "possible_explanation": "Why they might disagree"
  }],
  "novel_insights": ["Insights that emerge from combining artifacts that aren't in any single one"],
  "gaps": ["Knowledge gaps revealed by the combination"],
  "confidence": 0.8,
  "method_rankings": {
    "metric_name": ["method1", "method2"]
  }
}

Output ONLY valid JSON."""

        user_msg = f"Artifacts:\n{all_context}\n\nProgrammatic analysis:\n{json.dumps(programmatic_analysis, indent=2, default=str)}"
        if question:
            user_msg += f"\n\nGuiding question: {question}"

        response = self._call_claude(system=system, user=user_msg)
        result = self._extract_json(response)
        result["programmatic_analysis"] = programmatic_analysis
        return result

    def _programmatic_compose(self, artifacts: list[Artifact]) -> dict[str, Any]:
        """Perform programmatic cross-artifact analysis."""
        analysis: dict[str, Any] = {
            "num_artifacts": len(artifacts),
            "artifact_types": [],
            "all_claims": [],
            "all_methods": set(),
            "metric_overlap": {},
        }

        for i, art in enumerate(artifacts):
            if isinstance(art, ExperimentResultArtifact):
                analysis["artifact_types"].append(f"{i}: experiment_result")
                for claim in art.claims:
                    analysis["all_claims"].append({
                        "artifact": i,
                        "statement": claim.statement,
                        "status": claim.status.value,
                        "confidence": claim.confidence,
                    })
                for result in art.results:
                    analysis["all_methods"].add(result.method_name)
                    for metric in result.metrics:
                        key = metric.name
                        if key not in analysis["metric_overlap"]:
                            analysis["metric_overlap"][key] = []
                        analysis["metric_overlap"][key].append({
                            "artifact": i,
                            "method": result.method_name,
                            "value": metric.value,
                        })

            elif isinstance(art, MethodComparisonArtifact):
                analysis["artifact_types"].append(f"{i}: method_comparison")
                for claim in art.claims:
                    analysis["all_claims"].append({
                        "artifact": i,
                        "statement": claim.statement,
                        "status": claim.status.value,
                        "confidence": claim.confidence,
                    })
                for method in art.methods:
                    analysis["all_methods"].add(method.name)

        analysis["all_methods"] = list(analysis["all_methods"])
        return analysis

    # ------------------------------------------------------------------
    # diff() — Compare two artifacts
    # ------------------------------------------------------------------

    def diff(
        self, artifact_a: Artifact, artifact_b: Artifact
    ) -> dict[str, Any]:
        """Detect differences and contradictions between two artifacts.

        This is critical for research: identifying when two papers disagree,
        when results don't replicate, or when claims are contradictory.

        Args:
            artifact_a: First artifact
            artifact_b: Second artifact

        Returns:
            Structured diff with contradictions, agreements, and unique findings
        """
        # Programmatic diff first
        programmatic_diff = self._programmatic_diff(artifact_a, artifact_b)

        # Then LLM-augmented deep diff
        context_a = self._artifact_to_context(artifact_a)
        context_b = self._artifact_to_context(artifact_b)

        system = """You are a research artifact comparison engine. You must identify
all differences, contradictions, agreements, and complementary findings
between two research artifacts.

Respond with a JSON object:
{
  "summary": "High-level summary of the comparison",
  "agreements": [{
    "topic": "What they agree on",
    "description": "Details of agreement",
    "confidence": 0.9
  }],
  "contradictions": [{
    "topic": "What they disagree on",
    "artifact_a_claim": "What artifact A says",
    "artifact_b_claim": "What artifact B says",
    "severity": "major|minor|nuanced",
    "possible_explanation": "Why they might disagree"
  }],
  "unique_to_a": ["Findings only in artifact A"],
  "unique_to_b": ["Findings only in artifact B"],
  "complementary": ["How the artifacts complement each other"],
  "methodology_differences": ["Differences in experimental setup"],
  "recommendation": "How to reconcile or use both artifacts"
}

Output ONLY valid JSON."""

        response = self._call_claude(
            system=system,
            user=f"Artifact A:\n{context_a}\n\nArtifact B:\n{context_b}\n\nProgrammatic diff:\n{json.dumps(programmatic_diff, indent=2, default=str)}",
        )
        result = self._extract_json(response)
        result["programmatic_diff"] = programmatic_diff
        return result

    def _programmatic_diff(
        self, a: Artifact, b: Artifact
    ) -> dict[str, Any]:
        """Compute programmatic differences between two artifacts."""
        diff: dict[str, Any] = {
            "type_a": type(a).__name__,
            "type_b": type(b).__name__,
            "same_type": type(a).__name__ == type(b).__name__,
        }

        # Compare methods overlap
        methods_a: set[str] = set()
        methods_b: set[str] = set()

        if isinstance(a, ExperimentResultArtifact):
            methods_a = {r.method_name for r in a.results}
        elif isinstance(a, MethodComparisonArtifact):
            methods_a = {m.name for m in a.methods}

        if isinstance(b, ExperimentResultArtifact):
            methods_b = {r.method_name for r in b.results}
        elif isinstance(b, MethodComparisonArtifact):
            methods_b = {m.name for m in b.methods}

        diff["shared_methods"] = list(methods_a & methods_b)
        diff["only_in_a"] = list(methods_a - methods_b)
        diff["only_in_b"] = list(methods_b - methods_a)

        # Compare metrics for shared methods
        if isinstance(a, ExperimentResultArtifact) and isinstance(b, ExperimentResultArtifact):
            metric_diffs = []
            for method_name in diff["shared_methods"]:
                result_a = next((r for r in a.results if r.method_name == method_name), None)
                result_b = next((r for r in b.results if r.method_name == method_name), None)
                if result_a and result_b:
                    metrics_a = {m.name: m.value for m in result_a.metrics}
                    metrics_b = {m.name: m.value for m in result_b.metrics}
                    shared_metrics = set(metrics_a.keys()) & set(metrics_b.keys())
                    for metric_name in shared_metrics:
                        val_a = metrics_a[metric_name]
                        val_b = metrics_b[metric_name]
                        if val_a != val_b:
                            metric_diffs.append({
                                "method": method_name,
                                "metric": metric_name,
                                "value_a": val_a,
                                "value_b": val_b,
                                "delta": val_b - val_a,
                                "relative_delta_pct": (
                                    round((val_b - val_a) / val_a * 100, 2)
                                    if val_a != 0 else None
                                ),
                            })
            diff["metric_differences"] = metric_diffs

        # Compare claims
        claims_a = []
        claims_b = []
        if hasattr(a, "claims"):
            claims_a = [c.statement for c in a.claims]
        if hasattr(b, "claims"):
            claims_b = [c.statement for c in b.claims]
        diff["num_claims_a"] = len(claims_a)
        diff["num_claims_b"] = len(claims_b)

        return diff
