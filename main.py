"""LLM-Native Research Artifacts (LNRA)

A system for representing scientific knowledge in machine-readable formats
that AI agents can directly operate on, manipulate, and reason with.

Quick start:
    # Convert a paper
    uv run python demo/convert_paper.py

    # Query an artifact
    uv run python demo/query_artifact.py

    # Run full demo (convert, query, compose, diff)
    uv run python demo/run_full_demo.py

    # Run benchmark (traditional vs artifact format)
    uv run python demo/run_benchmark.py

    # Use as a library
    from lnra.converter import PaperConverter
    from lnra.agent import ArtifactAgent
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def quick_demo():
    """Quick demo using pre-built example artifacts (no API calls needed)."""
    from lnra.schemas.experiment import ExperimentResultArtifact
    from lnra.schemas.method_comparison import MethodComparisonArtifact

    examples_dir = Path(__file__).resolve().parent / "examples"

    # Load example experiment artifact
    exp_path = examples_dir / "example_experiment_artifact.json"
    exp_data = json.loads(exp_path.read_text())
    experiment = ExperimentResultArtifact.model_validate(exp_data)

    print("=" * 60)
    print("LLM-Native Research Artifacts - Quick Demo")
    print("=" * 60)

    print(f"\nLoaded: {experiment.metadata.title}")
    print(f"Type: {experiment.metadata.artifact_type.value}")
    print(f"Tags: {experiment.metadata.tags}")

    # Programmatic queries (no API needed)
    print("\n--- Programmatic Queries ---")

    # Best result
    best = experiment.get_best_result("BLEU")
    if best:
        bleu_metrics = [m for m in best.metrics if m.name == "BLEU"]
        print(f"\nBest BLEU: {best.method_name}")
        for m in bleu_metrics:
            print(f"  {m.dataset}: {m.value}")

    # Claims
    claims = experiment.get_claims_with_evidence()
    print(f"\nClaims ({len(claims)}):")
    for c in claims:
        print(f"  [{c['status']}] {c['claim'][:70]}...")
        print(f"    Confidence: {c['confidence']}")

    # Key findings
    print(f"\nKey Findings:")
    for f in experiment.key_findings:
        print(f"  - {f}")

    # Failure cases
    print(f"\nFailure Cases:")
    for f in experiment.failure_cases:
        print(f"  - {f}")

    # Load method comparison
    mc_path = examples_dir / "example_method_comparison.json"
    mc_data = json.loads(mc_path.read_text())
    comparison = MethodComparisonArtifact.model_validate(mc_data)

    print(f"\n\nLoaded: {comparison.metadata.title}")

    # Method profiles
    for method in comparison.methods:
        profile = comparison.get_method_profile(method.id)
        print(f"\n  {method.name} ({method.category.value}):")
        print(f"    Innovation: {method.key_innovation}")
        if profile.get("scores"):
            print(f"    Scores: {profile['scores']}")
        if method.preconditions:
            print(f"    Preconditions:")
            for p in method.preconditions:
                print(f"      - {p.description}")

    print(f"\n  Recommendation: {comparison.recommendation}")

    print("\n" + "=" * 60)
    print("This was a programmatic demo using pre-built artifacts.")
    print("For the full demo with Claude API conversion, run:")
    print("  uv run python demo/run_full_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    quick_demo()
