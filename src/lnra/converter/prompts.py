"""Prompts for the converter pipeline.

Each prompt instructs Claude to extract structured data from paper text
and output it as JSON conforming to our Pydantic schemas.
"""

ARTIFACT_TYPE_DETECTION_PROMPT = """You are a scientific paper analyzer. Given the beginning of a paper,
determine what type of structured artifact best represents its content.

Choose ONE of:
- experiment_result: The paper primarily reports experimental results (benchmarks, evaluations, ablation studies)
- method_comparison: The paper primarily compares multiple methods/approaches (surveys, comparative studies)

Respond with ONLY the artifact type name, nothing else."""


EXPERIMENT_EXTRACTION_PROMPT = """You are an expert scientific data extractor. Your job is to convert
a research paper into a structured JSON artifact that AI agents can directly reason with.

Output a single JSON object conforming to this structure:

{
  "metadata": {
    "title": "Paper title",
    "description": "1-2 sentence summary",
    "tags": ["tag1", "tag2"],
    "domain": "machine_learning",
    "provenance": [{
      "source_type": "paper",
      "source_id": "DOI or arXiv ID if available",
      "source_title": "Full paper title",
      "extraction_method": "claude-structured-extraction",
      "confidence": 0.85
    }]
  },
  "setup": {
    "task": "What task is being evaluated",
    "methodology": "How the experiment was conducted",
    "datasets": [{
      "name": "dataset name",
      "size": null,
      "splits": {"train": 0, "test": 0},
      "preprocessing": [],
      "characteristics": {}
    }],
    "hyperparameters": [{
      "name": "param name",
      "value": "value",
      "param_type": "continuous|discrete|categorical|boolean",
      "is_tuned": false
    }],
    "hardware": "compute info if mentioned",
    "num_runs": 1,
    "evaluation_protocol": "how results are evaluated",
    "baselines": ["baseline1", "baseline2"],
    "conditions": [{
      "condition_type": "assumption|precondition|limitation|scope",
      "description": "description",
      "is_verified": true
    }]
  },
  "results": [{
    "method_name": "Method name",
    "is_proposed": true,
    "metrics": [{
      "name": "metric name",
      "value": 0.95,
      "higher_is_better": true,
      "dataset": "which dataset",
      "split": "test",
      "uncertainty": {
        "uncertainty_type": "statistical",
        "value": 0.02,
        "description": "standard deviation over 3 runs"
      }
    }]
  }],
  "claims": [{
    "statement": "The claim in natural language",
    "status": "supported|contested|preliminary|established",
    "confidence": 0.8,
    "evidence_for": [{
      "evidence_type": "quantitative|experimental|observational",
      "description": "What evidence supports this",
      "strength": 0.8,
      "data": {}
    }],
    "evidence_against": [],
    "conditions": [{
      "condition_type": "scope|limitation",
      "description": "When this claim holds"
    }]
  }],
  "causal_relationships": [{
    "cause": "factor",
    "effect": "outcome",
    "strength": "strong|moderate|weak|suggested|correlational",
    "mechanism": "proposed mechanism",
    "conditions": [],
    "confounders": []
  }],
  "ablation_results": [],
  "failure_cases": ["Known failure case 1"],
  "key_findings": ["Finding 1", "Finding 2"]
}

IMPORTANT RULES:
1. Extract ALL quantitative results from tables and text
2. Be explicit about uncertainty — if std dev or confidence intervals are reported, include them
3. Identify ALL conditions and limitations mentioned in the paper
4. Separate claims from evidence — a claim is an interpretation, evidence is data
5. If something is not mentioned in the paper, use null or empty arrays, don't fabricate
6. Include causal claims explicitly — many papers imply causation from correlation
7. Note any ablation studies separately
8. Confidence scores should reflect your assessment of extraction accuracy

Output ONLY valid JSON, no other text."""


METHOD_COMPARISON_EXTRACTION_PROMPT = """You are an expert scientific data extractor. Your job is to convert
a research paper that compares methods into a structured JSON artifact.

Output a single JSON object conforming to this structure:

{
  "metadata": {
    "title": "Paper title",
    "description": "1-2 sentence summary",
    "tags": ["tag1", "tag2"],
    "domain": "machine_learning",
    "provenance": [{
      "source_type": "paper",
      "source_id": "DOI or arXiv ID",
      "source_title": "Full paper title",
      "extraction_method": "claude-structured-extraction",
      "confidence": 0.85
    }]
  },
  "methods": [{
    "name": "Method name",
    "category": "proposed|baseline|state_of_the_art|ablation|oracle",
    "description": "Brief description",
    "key_innovation": "What makes this method unique",
    "preconditions": [{
      "condition_type": "precondition",
      "description": "What must be true for this to work"
    }],
    "limitations": ["limitation 1"],
    "computational_cost": "relative cost info",
    "paper_reference": "citation",
    "year": 2024,
    "components": ["component1", "component2"]
  }],
  "dimensions": [{
    "name": "accuracy",
    "description": "Classification accuracy",
    "unit": "%",
    "higher_is_better": true,
    "weight": 1.0
  }],
  "results": [{
    "dimension": {
      "name": "accuracy",
      "higher_is_better": true
    },
    "scores": [{
      "method_id": "will be matched by position",
      "dimension_name": "accuracy",
      "value": 95.2,
      "uncertainty": null,
      "notes": null
    }],
    "winner_id": null,
    "statistical_significance": null
  }],
  "claims": [{
    "statement": "Claim text",
    "status": "supported|contested|preliminary",
    "confidence": 0.8,
    "evidence_for": [],
    "evidence_against": [],
    "conditions": []
  }],
  "tradeoffs": [{
    "description": "Method A is faster but less accurate than Method B",
    "methods_involved": ["method_id_1", "method_id_2"],
    "dimensions_involved": ["speed", "accuracy"],
    "recommendation": "Use Method A for real-time applications"
  }],
  "recommendation": "Overall recommendation"
}

IMPORTANT RULES:
1. Extract ALL methods mentioned, including baselines
2. Identify all comparison dimensions (metrics, cost, memory, etc.)
3. Be explicit about preconditions — what does each method need to work?
4. Identify tradeoffs — when is one method better than another?
5. Note computational costs when available
6. Separate claims from evidence
7. Include uncertainty when reported
8. If data is not available, use null, don't fabricate

Output ONLY valid JSON, no other text."""
