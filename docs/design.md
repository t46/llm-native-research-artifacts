# Design Document: LLM-Native Research Artifacts

## 1. Motivation

Scientific knowledge today is locked in human-readable formats: PDF papers, PNG figures, LaTeX equations. AI agents must "read" these documents just like humans do — parsing natural language, interpreting tables, and extracting meaning from unstructured text. This is fundamentally inefficient for AI-to-AI scientific communication.

**The core insight**: If AI agents are increasingly the primary producers and consumers of scientific knowledge, we need a **machine-native representation** — structured, queryable, composable artifacts that AI agents can directly operate on.

## 2. Knowledge Granularity Model

We propose a four-level granularity model:

```
Level 4: Paper          — Full document (current standard)
Level 3: Snapshot       — Point-in-time research state (ARH concept)
Level 2: Claim/Evidence — Individual assertions with backing data
Level 1: Structured Artifact — Machine-readable, typed, validated data
```

This project operates primarily at **Level 1**, creating artifacts that decompose papers into their constituent parts with explicit types, relationships, and metadata.

### Comparison with traditional formats

| Aspect | Paper (PDF) | LLM-Native Artifact |
|--------|------------|-------------------|
| Format | Natural language + figures | Typed JSON/YAML with Pydantic validation |
| Queryability | Full-text search only | Programmatic field access + semantic query |
| Composability | Manual reading + synthesis | Automatic cross-artifact operations |
| Uncertainty | Implicit in prose | Explicit `UncertaintyEstimate` objects |
| Conditions | Buried in text | First-class `Condition` objects |
| Provenance | Citations | Machine-trackable `Provenance` chain |
| Causation | Implied | Explicit `CausalRelationship` objects |

## 3. Two-Layer Communication Model

```
┌─────────────────────────────────┐
│     AI-to-AI Layer (Fast)       │
│  Structured artifacts, JSON     │
│  Programmatic query/compose     │
│  Machine-validated schemas      │
│  Direct field access            │
└──────────────┬──────────────────┘
               │ Translation
┌──────────────▼──────────────────┐
│   AI-to-Human Layer (Readable)  │
│  Natural language summaries     │
│  Visualizations, tables         │
│  Explanations of uncertainty    │
│  Narrative framing              │
└─────────────────────────────────┘
```

The AI-to-AI layer is the **primary representation**. Human-readable outputs are generated on demand by translating from the structured artifact. This inverts the current paradigm where human-readable papers are primary and machine extraction is secondary.

## 4. Artifact Schema Design

### Core Principles

1. **Explicit uncertainty**: Every claim and measurement carries explicit uncertainty information
2. **Conditional knowledge**: Claims are tagged with the conditions under which they hold
3. **Provenance tracking**: Every piece of data tracks where it came from and how it was extracted
4. **Causal claims as first-class citizens**: Papers often imply causation; we make it explicit
5. **Evidence-claim separation**: Claims are distinct from the evidence supporting them

### Artifact Types

**ExperimentResultArtifact**: Represents experimental results from a paper
- Setup (task, datasets, hyperparameters, hardware)
- Results (per-method metrics with uncertainty)
- Claims (with evidence for/against)
- Causal relationships
- Ablation results
- Failure cases

**MethodComparisonArtifact**: Represents a comparison between methods
- Method descriptions (with preconditions, limitations)
- Comparison dimensions
- Scored results per dimension
- Tradeoff analysis
- Recommendations

### Schema Validation

All artifacts are validated via Pydantic v2 models, ensuring:
- Type correctness at all levels
- Required fields are present
- Value ranges are respected (e.g., confidence in [0, 1])
- Enum values are valid

## 5. Agent Interface Design

The three core operations:

### query(artifact, question) -> structured_answer

Two modes:
1. **Programmatic**: Common questions (best result, preconditions, claims) are answered directly from the artifact structure without LLM calls
2. **LLM-augmented**: Complex questions are answered by passing the artifact as context to Claude, which reasons over the structured data

The key advantage: even LLM-augmented queries are more precise because the artifact has explicit structure for conditions, uncertainty, claims, etc.

### compose(artifacts, question?) -> synthesis

Combines multiple artifacts to find:
- Shared findings across papers
- Contradictions between papers
- Novel insights from combination
- Knowledge gaps

This is the killer feature: cross-paper analysis that would take a human researcher hours becomes a single operation.

### diff(artifact_a, artifact_b) -> differences

Identifies:
- Agreements and contradictions
- Methodology differences
- Complementary findings
- Shared vs unique methods

Critical for reproducibility research and meta-analysis.

## 6. Relationship to ARH (AI Researcher Hub)

ARH and LNRA are complementary:

```
ARH (AI Researcher Hub)                LNRA (This Project)
├── Research Projects                  ├── Artifact Schemas
├── Snapshots (Markdown papers)  ←→    ├── Converter Pipeline
├── Reviews & Comments                 ├── Agent Interface
├── Agent collaboration                ├── Cross-artifact operations
└── Human-readable focus               └── Machine-readable focus
```

### Integration Possibilities

1. **ARH Snapshots → LNRA Artifacts**: When an agent publishes a snapshot in ARH, automatically generate LNRA artifacts for machine consumption
2. **LNRA Query API for ARH**: ARH agents could query the structured artifact layer for precise answers instead of reading snapshot text
3. **LNRA Compose for Reviews**: When reviewing a snapshot, use compose() to automatically compare against related work
4. **Bidirectional sync**: Changes to the artifact update the human-readable snapshot and vice versa

### Knowledge Flow

```
Paper (PDF) → [Converter] → LNRA Artifact → [ARH Integration] → Snapshot
                                  ↕
                            Agent Operations
                         (query, compose, diff)
```

## 7. Benchmark Design

The benchmark compares AI agent performance on the same research questions using:
- **(a) Traditional**: Paper text given as context
- **(b) Artifact**: Structured artifact given as context

Evaluation dimensions:
1. **Accuracy**: Is the answer factually correct?
2. **Specificity**: Does it include specific numbers and conditions?
3. **Completeness**: Does it cover all relevant aspects?
4. **Groundedness**: Is it clearly backed by evidence?

The hypothesis: structured artifacts lead to more specific, grounded answers because the data is already organized for programmatic access.

## 8. Future Directions

1. **More artifact types**: Literature review, hypothesis, dataset description
2. **Artifact evolution**: Track how artifacts change over time (version diffs)
3. **Knowledge graphs**: Link artifacts into a queryable graph of scientific knowledge
4. **Automatic validation**: Cross-reference claims across artifacts automatically
5. **Domain-specific schemas**: Extend base schemas for biology, physics, etc.
6. **Real-time extraction**: Convert papers to artifacts as they're published
7. **Collaborative artifacts**: Multiple agents contributing to the same artifact
