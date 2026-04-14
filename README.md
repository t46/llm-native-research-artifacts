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

## Autoresearch Validation (2026-04-14)

LNRA has been validated against real autoresearch pipeline outputs -- not just manually
curated paper text. See [data/autoresearch_artifacts/validation_report.md](data/autoresearch_artifacts/validation_report.md)
for the full report.

### Data Sources Tested

| Source | Type | Size | Result |
|--------|------|------|--------|
| Vanilla autoresearch REPORT.md | Japanese system design doc | 18K chars | Converted (after bug fix) |
| Auto-research-evaluator (AI Scientist-v2, 2 runs) | English evaluation reports | 41-49K chars | Converted |
| Auto-research-evaluator (D-Separation) | English evaluation report | 39K chars | Converted |

### Key Findings

1. **Converter works on autoresearch outputs** with minor fixes. The main issue was
   provenance validation -- autoresearch outputs generate non-standard `source_type` values
   and null `source_id` fields. Fixed in `_repair_data()`.

2. **query() is effective**: Both programmatic (instant, structured lookups) and LLM-augmented
   (interpretive, contextual) query paths work on autoresearch artifacts. 16/16 queries
   successful.

3. **compose() reveals cross-artifact insights**: Synthesizing autoresearch artifacts surfaced
   patterns (critique calibration as a key differentiator, d-separation explaining ablation gaps)
   not visible in individual artifacts.

4. **diff() detects meaningful differences**: Successfully found agreements and contradictions
   between evaluations of the same paper, and complementary findings across different topics.

5. **Autoresearch-specific challenges**: Schema-data mismatch (system docs vs paper format),
   provenance chain complexity, quantitative sparsity, and mixed-language content require
   targeted improvements.

### Validation Script

```bash
# Run the full validation (requires ANTHROPIC_API_KEY)
uv run python scripts/validate_autoresearch.py
```

## Karpathy-style Autoresearch Validation (2026-04-14)

LNRA has been validated against a Karpathy-style iterative experiment loop:
an autonomous agent running 20 CIFAR-10 experiments via autoresearch-lite,
producing a `results.tsv` experiment log, final `train.py`, and git history.
This is structurally very different from papers -- it is tabular experiment
data with code, not narrative text.

See [data/karpathy_artifacts/validation_report.md](data/karpathy_artifacts/validation_report.md)
for the full report.

### Adapter Design

An adapter assembles the full autoresearch session (results.tsv + train.py +
program.md + git log) into a single pseudo-paper document that the existing
converter can process. This tests the converter's robustness on non-paper inputs.

### Data

| Component | Description |
|-----------|-------------|
| results.tsv | 21 experiments: commit, val_accuracy, memory_gb, status, description |
| train.py | Final best model (73.99% accuracy) |
| program.md | Agent instructions (Karpathy-style protocol) |
| git log | 4 commits (baseline + 3 kept improvements) |

### Results

| Operation | Status | Details |
|-----------|--------|---------|
| Convert | Success | 3 results, 3 claims, 6 key findings, 2 causal relationships, 5 failure cases |
| Query (5 questions) | 5/5 success | Confidence 0.9-0.95, all LLM-augmented |
| Compose (with Attention artifact) | Success | 6 shared findings, 4 novel insights, 2 contradictions |
| Diff (vs Attention artifact) | Success | 2 agreements, 2 contradictions, 5 methodology differences |

### Key Findings

1. **Converter handles experiment logs**: The tabular format (TSV with keep/discard/crash
   status) was successfully converted to structured claims and results. The adapter
   transforms iterative experiment data into a format the converter understands.

2. **Query answers are accurate and grounded**: All 5 domain-specific questions (most
   effective change, hyperparameter sensitivity, crash analysis, strategy emergence,
   diminishing returns) received well-grounded answers with evidence and caveats.

3. **Cross-domain compose works**: Composing a CIFAR-10 CNN optimization session with
   the Attention Is All You Need artifact surfaced novel insights about iterative
   optimization vs architectural innovation as complementary research strategies.

4. **Diff detects structural differences**: The programmatic diff correctly identified
   zero method overlap (CNN configurations vs Transformer variants), while the
   LLM-augmented diff found meaningful methodology and paradigm differences.

5. **Crash data preserved as failure cases**: The two crashes (residual connection
   architecture error, LLM parsing error) were correctly captured as failure cases
   in the artifact.

### Validation Script

```bash
# Run the Karpathy-style validation (requires ANTHROPIC_API_KEY)
uv run python scripts/validate_karpathy.py
```

## Technical Details

- **Python 3.13+** with `uv` for package management
- **Pydantic v2** for schema validation
- **Anthropic Claude API** for paper conversion and LLM-augmented queries
- **JSON** as the artifact serialization format

## License

MIT
