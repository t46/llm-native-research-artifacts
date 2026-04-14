# LLM-Native Research Artifacts

A system for representing scientific knowledge in **machine-readable formats** that AI agents can directly operate on, manipulate, and reason with.

Instead of papers (PDF) and figures (PNG), scientific knowledge becomes **typed, validated, queryable artifacts** with explicit uncertainty, conditions, provenance, and causal relationships.

## Why This Exists

Scientific communication is designed for humans: natural language papers, visual figures, implicit assumptions. As AI agents become primary producers and consumers of research, we need a **machine-native representation layer** where:

- **Uncertainty is explicit**: Every measurement carries an `UncertaintyEstimate`, not a parenthetical "(+/- 0.5)"
- **Conditions are first-class**: "This works when X > Y" is a typed `Condition`, not a sentence in section 4.3
- **Provenance is tracked**: Every data point traces back to its source through a `Provenance` chain
- **Causation is declared**: "A causes B" is a `CausalRelationship` with strength and confounders, not an implied conclusion
- **Cross-paper analysis is programmatic**: Comparing two papers becomes `agent.diff(artifact_a, artifact_b)`

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Papers     │────▶│  Converter   │────▶│  Artifacts   │
│  (text/PDF)  │     │  (Claude)    │     │  (JSON)      │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                    ┌─────────────▼─────────────┐
                                    │     Agent Interface       │
                                    │  query() compose() diff() │
                                    └───��───────────────────────┘
```

### Components

1. **Artifact Schemas** (`src/lnra/schemas/`): Pydantic v2 models for structured research data
   - `ExperimentResultArtifact`: Experimental results with setup, metrics, claims, ablations
   - `MethodComparisonArtifact`: Method comparisons with dimensions, tradeoffs, recommendations

2. **Converter Pipeline** (`src/lnra/converter/`): Claude-powered extraction from paper text to structured artifacts

3. **Agent Interface** (`src/lnra/agent/`): Three core operations for AI agents:
   - `query()`: Ask questions, get structured answers with evidence and confidence
   - `compose()`: Synthesize insights across multiple artifacts
   - `diff()`: Find contradictions and differences between artifacts

4. **Benchmark** (`src/lnra/benchmark/`): Compare AI performance using traditional text vs structured artifacts

## Quick Start

```bash
# Install dependencies
uv sync

# Run the quick demo (no API key needed, uses pre-built artifacts)
uv run python main.py

# Convert a paper to an artifact (requires ANTHROPIC_API_KEY)
uv run python demo/convert_paper.py

# Query an artifact
uv run python demo/query_artifact.py

# Full demo: convert, query, compose, diff
uv run python demo/run_full_demo.py

# Benchmark: traditional vs artifact format
uv run python demo/run_benchmark.py
```

## Example: Programmatic Research Queries

```python
from lnra.schemas.experiment import ExperimentResultArtifact
from lnra.agent import ArtifactAgent
import json

# Load an artifact
artifact = ExperimentResultArtifact.model_validate(json.loads(open("artifact.json").read()))

# Programmatic access (no LLM needed)
best = artifact.get_best_result("BLEU")
claims = artifact.get_claims_with_evidence()
comparison = artifact.get_metric_comparison("accuracy")

# Agent-powered queries
agent = ArtifactAgent()
answer = agent.query(artifact, "What are the preconditions of this method?")
# Returns: {"answer": "...", "evidence": [...], "confidence": 0.9, "caveats": [...]}

# Cross-paper synthesis
insights = agent.compose([artifact1, artifact2], "What patterns emerge?")

# Find contradictions
diff = agent.diff(artifact1, artifact2)
```

## Artifact Schema

Every artifact contains:

```
metadata        — ID, type, title, tags, provenance, schema version
├── provenance  — Where the data came from (paper DOI, extraction method, confidence)
claims          — Scientific claims with evidence for/against
├── evidence    — Typed evidence (quantitative, experimental, theoretical)
├── conditions  — When the claim holds (preconditions, assumptions, scope)
├── uncertainty — Explicit uncertainty (statistical, systematic, epistemic)
causal_relationships — Explicit causal claims with mechanism and confounders
```

### Experiment Result Artifact
```
setup           — Task, methodology, datasets, hyperparameters, hardware
results         — Per-method metrics with uncertainty and conditions
ablation_results — Ablation study results
failure_cases   — Known failure cases
key_findings    — Summary findings
```

### Method Comparison Artifact
```
methods         — Method descriptions with preconditions and limitations
dimensions      — Comparison dimensions (accuracy, speed, memory, etc.)
results         — Scores per method per dimension
tradeoffs       — Explicit tradeoff analysis
recommendation  — Overall recommendation
```

## Connection to Metascience

This project is a **metascience experiment** exploring the question: _What if scientific knowledge was structured for machines first, and translated for humans second?_

Key hypotheses being tested:
1. Structured artifacts enable more precise and grounded answers than raw paper text
2. Cross-paper analysis (compose, diff) reveals insights that individual paper reading misses
3. Explicit representation of uncertainty, conditions, and causation reduces misinterpretation

See [docs/design.md](docs/design.md) for the full design philosophy and the two-layer communication model.

## Relationship to AI Researcher Hub (ARH)

This project is complementary to [AI Researcher Hub](https://github.com/t46/ai-researcher-hub):

- **ARH**: Human-readable research platform (snapshots, reviews, collaboration)
- **LNRA**: Machine-readable research artifacts (structured data, programmatic operations)

Future integration: ARH snapshots could automatically generate LNRA artifacts, and LNRA's query/compose/diff operations could power ARH's review and discovery features.

## Technical Details

- **Python 3.13+** with `uv` for package management
- **Pydantic v2** for schema validation
- **Anthropic Claude API** for paper conversion and LLM-augmented queries
- **JSON** as the artifact serialization format

## License

MIT
