"""Converter pipeline using Claude API to extract structured data from papers.

Input: paper text, figure descriptions, experimental results
Output: LLM-native artifact in the defined schema
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from ..schemas.base import ArtifactType
from ..schemas.experiment import ExperimentResultArtifact
from ..schemas.method_comparison import MethodComparisonArtifact
from .prompts import (
    EXPERIMENT_EXTRACTION_PROMPT,
    METHOD_COMPARISON_EXTRACTION_PROMPT,
    ARTIFACT_TYPE_DETECTION_PROMPT,
)

logger = logging.getLogger(__name__)


class PaperConverter:
    """Converts paper text into LLM-native research artifacts using Claude.

    Usage:
        converter = PaperConverter()
        artifact = converter.convert(paper_text)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
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
        """Extract JSON from Claude's response, handling markdown code blocks."""
        # Try to find JSON in code blocks first
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        return json.loads(text)

    @staticmethod
    def _repair_data(data: dict[str, Any]) -> dict[str, Any]:
        """Repair common issues in LLM-generated JSON to match our schemas.

        LLMs sometimes produce outputs that don't exactly match Pydantic models:
        - Strings where Condition objects are expected
        - Non-integer values in dict[str, int] fields
        - Missing required fields like method_name in ablation results
        - Invalid enum values
        """
        VALID_UNCERTAINTY_TYPES = {"statistical", "systematic", "measurement", "model", "epistemic", "aleatoric"}
        VALID_CONDITION_TYPES = {"precondition", "assumption", "constraint", "limitation", "scope", "environment"}
        VALID_EVIDENCE_TYPES = {"quantitative", "qualitative", "experimental", "observational", "theoretical", "simulation", "meta_analysis"}
        VALID_CAUSAL_STRENGTHS = {"strong", "moderate", "weak", "suggested", "correlational"}
        VALID_CLAIM_STATUSES = {"supported", "contested", "refuted", "preliminary", "established"}
        VALID_SOURCE_TYPES = {"paper", "dataset", "code", "human_annotation", "llm_extraction", "llm_synthesis", "experiment", "derived"}

        def _fix_enum_recursive(obj: Any) -> Any:
            """Walk the data recursively and fix known enum fields."""
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if k == "uncertainty_type" and isinstance(v, str) and v not in VALID_UNCERTAINTY_TYPES:
                        obj[k] = "epistemic"  # safe default
                    elif k == "condition_type" and isinstance(v, str) and v not in VALID_CONDITION_TYPES:
                        obj[k] = "scope"
                    elif k == "evidence_type" and isinstance(v, str) and v not in VALID_EVIDENCE_TYPES:
                        obj[k] = "qualitative"
                    elif k == "strength" and isinstance(v, str) and v not in VALID_CAUSAL_STRENGTHS:
                        obj[k] = "moderate"
                    elif k == "status" and isinstance(v, str) and v not in VALID_CLAIM_STATUSES:
                        obj[k] = "preliminary"
                    elif k == "param_type" and isinstance(v, str) and v not in {"continuous", "discrete", "categorical", "boolean"}:
                        obj[k] = "continuous"
                    elif k == "category" and isinstance(v, str) and v not in {"proposed", "baseline", "state_of_the_art", "ablation", "oracle"}:
                        obj[k] = "baseline"
                    elif k == "source_type" and isinstance(v, str) and v not in VALID_SOURCE_TYPES:
                        obj[k] = "derived"  # safe default for non-standard source types
                    else:
                        obj[k] = _fix_enum_recursive(v)
                return obj
            elif isinstance(obj, list):
                return [_fix_enum_recursive(item) for item in obj]
            return obj

        data = _fix_enum_recursive(data)

        def _fix_conditions(items: list) -> list:
            """Convert plain strings to Condition dicts."""
            fixed = []
            for item in items:
                if isinstance(item, str):
                    fixed.append({
                        "condition_type": "scope",
                        "description": item,
                    })
                elif isinstance(item, dict):
                    fixed.append(item)
                else:
                    fixed.append(item)
            return fixed

        def _fix_splits(splits: dict | None) -> dict:
            """Ensure split values are integers, drop non-integer entries."""
            if not splits:
                return {}
            fixed = {}
            for k, v in splits.items():
                if isinstance(v, int):
                    fixed[k] = v
                elif isinstance(v, float):
                    fixed[k] = int(v)
                elif isinstance(v, str):
                    try:
                        fixed[k] = int(v)
                    except ValueError:
                        pass  # Drop non-numeric split entries
                # Skip None values
            return fixed

        def _fix_ablation_results(results: list) -> list:
            """Ensure ablation results have method_name field."""
            fixed = []
            for r in results:
                if isinstance(r, dict):
                    if "method_name" not in r:
                        # Try to get a name from other fields
                        name = (
                            r.pop("component", None)
                            or r.pop("name", None)
                            or r.pop("variant", None)
                            or r.pop("configuration", None)
                            or "unnamed_ablation"
                        )
                        r["method_name"] = name
                    fixed.append(r)
            return fixed

        # Walk the data and fix known issues
        # Fix provenance entries (null source_id, invalid source_type)
        def _fix_provenance_list(provenance_list: list) -> list:
            """Ensure provenance entries have valid required fields."""
            for prov in provenance_list:
                if isinstance(prov, dict):
                    if prov.get("source_id") is None:
                        prov["source_id"] = "unknown"
                    if prov.get("source_type") not in VALID_SOURCE_TYPES:
                        prov["source_type"] = "derived"
            return provenance_list

        if "metadata" in data and "provenance" in data["metadata"]:
            if isinstance(data["metadata"]["provenance"], list):
                data["metadata"]["provenance"] = _fix_provenance_list(data["metadata"]["provenance"])

        # Fix setup.datasets[].splits
        if "setup" in data and "datasets" in data["setup"]:
            for ds in data["setup"]["datasets"]:
                if isinstance(ds, dict) and "splits" in ds:
                    ds["splits"] = _fix_splits(ds["splits"])

        # Fix setup.conditions
        if "setup" in data and "conditions" in data["setup"]:
            data["setup"]["conditions"] = _fix_conditions(data["setup"]["conditions"])

        # Fix causal_relationships[].conditions
        if "causal_relationships" in data:
            for cr in data["causal_relationships"]:
                if isinstance(cr, dict) and "conditions" in cr:
                    cr["conditions"] = _fix_conditions(cr["conditions"])

        # Fix claims[].conditions and claims[].evidence_for/against
        if "claims" in data:
            for claim in data["claims"]:
                if isinstance(claim, dict):
                    if "conditions" in claim:
                        claim["conditions"] = _fix_conditions(claim["conditions"])
                    for key in ("evidence_for", "evidence_against"):
                        if key in claim and isinstance(claim[key], list):
                            for ev in claim[key]:
                                if isinstance(ev, dict) and "conditions" in ev:
                                    ev["conditions"] = _fix_conditions(ev["conditions"])

        # Fix results[].metrics[].conditions
        if "results" in data:
            for result in data["results"]:
                if isinstance(result, dict) and "metrics" in result:
                    for metric in result["metrics"]:
                        if isinstance(metric, dict) and "conditions" in metric:
                            metric["conditions"] = _fix_conditions(metric["conditions"])

        # Fix ablation_results
        if "ablation_results" in data:
            data["ablation_results"] = _fix_ablation_results(data["ablation_results"])

        # Fix methods[].preconditions (for MethodComparison)
        if "methods" in data:
            for method in data["methods"]:
                if isinstance(method, dict) and "preconditions" in method:
                    method["preconditions"] = _fix_conditions(method["preconditions"])

        return data

    def detect_artifact_type(self, paper_text: str) -> ArtifactType:
        """Detect what type of artifact best represents this paper."""
        response = self._call_claude(
            system=ARTIFACT_TYPE_DETECTION_PROMPT,
            user=paper_text[:4000],  # Use first 4000 chars for detection
        )
        response_lower = response.lower().strip()
        if "method_comparison" in response_lower:
            return ArtifactType.METHOD_COMPARISON
        return ArtifactType.EXPERIMENT_RESULT

    def convert(
        self,
        paper_text: str,
        artifact_type: ArtifactType | None = None,
        title: str | None = None,
    ) -> ExperimentResultArtifact | MethodComparisonArtifact:
        """Convert paper text into an LLM-native artifact.

        Args:
            paper_text: Full text of the paper
            artifact_type: Type of artifact to create (auto-detected if None)
            title: Optional title override

        Returns:
            A structured artifact (ExperimentResult or MethodComparison)
        """
        if artifact_type is None:
            artifact_type = self.detect_artifact_type(paper_text)
            logger.info(f"Auto-detected artifact type: {artifact_type}")

        if artifact_type == ArtifactType.EXPERIMENT_RESULT:
            return self._convert_to_experiment(paper_text, title)
        elif artifact_type == ArtifactType.METHOD_COMPARISON:
            return self._convert_to_method_comparison(paper_text, title)
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

    def _convert_to_experiment(
        self, paper_text: str, title: str | None = None
    ) -> ExperimentResultArtifact:
        """Convert paper text to an ExperimentResultArtifact."""
        logger.info("Converting paper to ExperimentResultArtifact...")

        response = self._call_claude(
            system=EXPERIMENT_EXTRACTION_PROMPT,
            user=f"Convert the following paper into a structured experiment result artifact:\n\n{paper_text}",
        )

        data = self._extract_json(response)

        # Ensure metadata fields
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["artifact_type"] = "experiment_result"
        if title:
            data["metadata"]["title"] = title

        # Repair LLM output to match our schemas
        data = self._repair_data(data)

        artifact = ExperimentResultArtifact.model_validate(data)
        return artifact

    def _convert_to_method_comparison(
        self, paper_text: str, title: str | None = None
    ) -> MethodComparisonArtifact:
        """Convert paper text to a MethodComparisonArtifact."""
        logger.info("Converting paper to MethodComparisonArtifact...")

        response = self._call_claude(
            system=METHOD_COMPARISON_EXTRACTION_PROMPT,
            user=f"Convert the following paper into a structured method comparison artifact:\n\n{paper_text}",
        )

        data = self._extract_json(response)

        # Ensure metadata fields
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["artifact_type"] = "method_comparison"
        if title:
            data["metadata"]["title"] = title

        # Repair LLM output to match our schemas
        data = self._repair_data(data)

        artifact = MethodComparisonArtifact.model_validate(data)
        return artifact

    def convert_with_context(
        self,
        paper_text: str,
        additional_context: str | None = None,
        figure_descriptions: list[str] | None = None,
        artifact_type: ArtifactType | None = None,
    ) -> ExperimentResultArtifact | MethodComparisonArtifact:
        """Convert with additional context (figure descriptions, etc.)."""
        enriched_text = paper_text
        if figure_descriptions:
            enriched_text += "\n\n## Figure Descriptions\n"
            for i, desc in enumerate(figure_descriptions, 1):
                enriched_text += f"\nFigure {i}: {desc}\n"
        if additional_context:
            enriched_text += f"\n\n## Additional Context\n{additional_context}\n"

        return self.convert(enriched_text, artifact_type=artifact_type)
