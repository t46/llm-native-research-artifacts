"""Demo: Convert a paper to an LLM-native artifact.

Usage:
    uv run python demo/convert_paper.py [--paper PATH] [--output PATH]

If no paper is given, uses the sample "Attention Is All You Need" text.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lnra.converter import PaperConverter
from lnra.schemas.base import ArtifactType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert a paper to an LLM-native artifact")
    parser.add_argument("--paper", type=str, default=None, help="Path to paper text file")
    parser.add_argument("--output", type=str, default=None, help="Output path for artifact JSON")
    parser.add_argument(
        "--type",
        type=str,
        choices=["experiment_result", "method_comparison", "auto"],
        default="auto",
        help="Artifact type (default: auto-detect)",
    )
    args = parser.parse_args()

    # Load paper text
    if args.paper:
        paper_path = Path(args.paper)
    else:
        paper_path = Path(__file__).resolve().parent.parent / "data" / "sample_paper_1.txt"

    logger.info(f"Loading paper from: {paper_path}")
    paper_text = paper_path.read_text()

    # Determine artifact type
    artifact_type = None if args.type == "auto" else ArtifactType(args.type)

    # Convert
    converter = PaperConverter()
    logger.info("Converting paper to LLM-native artifact...")
    artifact = converter.convert(paper_text, artifact_type=artifact_type)

    # Output
    artifact_json = artifact.model_dump(mode="json")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).resolve().parent.parent / "data" / "artifact_output.json"

    output_path.write_text(json.dumps(artifact_json, indent=2, default=str))
    logger.info(f"Artifact saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Artifact Type: {artifact.metadata.artifact_type.value}")
    print(f"Title: {artifact.metadata.title}")
    print(f"Description: {artifact.metadata.description}")
    print(f"Tags: {artifact.metadata.tags}")
    print(f"Schema Version: {artifact.metadata.schema_version}")

    if hasattr(artifact, "results"):
        print(f"\nResults: {len(artifact.results)} method(s)")
        for r in artifact.results:
            metrics_str = ", ".join(f"{m.name}={m.value}" for m in r.metrics)
            print(f"  - {r.method_name}: {metrics_str}")

    if hasattr(artifact, "claims"):
        print(f"\nClaims: {len(artifact.claims)}")
        for c in artifact.claims:
            stmt = c.statement[:80]
            print(f"  - [{c.status.value}] {stmt}...")

    if hasattr(artifact, "key_findings"):
        print(f"\nKey Findings: {len(artifact.key_findings)}")
        for f in artifact.key_findings:
            print(f"  - {f[:80]}")

    print("=" * 60)


if __name__ == "__main__":
    main()
