"""LNRA Interactive Web Demo — Streamlit UI.

Explore, query, compose, and diff LLM-native research artifacts.
Launch with: uv run streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure the src/ package is importable (project is not pip-installed).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LNRA Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Helpers: artifact loading
# ---------------------------------------------------------------------------


def _find_artifact_files(root: Path) -> list[Path]:
    """Recursively find JSON files that look like LNRA artifacts."""
    files: list[Path] = []
    for p in sorted(root.rglob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict) and "metadata" in data and "claims" in data:
                files.append(p)
        except Exception:
            continue
    return files


@st.cache_data(show_spinner="Scanning data directory...")
def discover_artifacts() -> list[str]:
    """Return a list of relative paths (from DATA_DIR) to valid artifacts."""
    return [str(p.relative_to(DATA_DIR)) for p in _find_artifact_files(DATA_DIR)]


def load_artifact_json(rel_path: str) -> dict[str, Any]:
    """Load raw JSON for an artifact by its relative path under data/."""
    with open(DATA_DIR / rel_path) as f:
        return json.load(f)


def load_artifact_object(raw: dict[str, Any]):
    """Parse raw JSON into the appropriate Pydantic model."""
    from lnra.schemas.experiment import ExperimentResultArtifact
    from lnra.schemas.method_comparison import MethodComparisonArtifact

    artifact_type = raw.get("metadata", {}).get("artifact_type", "")
    if artifact_type == "experiment_result":
        return ExperimentResultArtifact.model_validate(raw)
    elif artifact_type == "method_comparison":
        return MethodComparisonArtifact.model_validate(raw)
    else:
        # Try experiment first, then comparison
        try:
            return ExperimentResultArtifact.model_validate(raw)
        except Exception:
            return MethodComparisonArtifact.model_validate(raw)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("LNRA Explorer")
st.sidebar.caption("LLM-Native Research Artifacts")

# Artifact selection
artifact_files = discover_artifacts()

if not artifact_files:
    st.sidebar.warning("No artifacts found in data/ directory.")
    st.stop()

# File upload
uploaded = st.sidebar.file_uploader(
    "Upload artifact JSON", type=["json"], accept_multiple_files=False
)
if uploaded is not None:
    try:
        uploaded_data = json.loads(uploaded.read())
        if "metadata" in uploaded_data and "claims" in uploaded_data:
            st.session_state["_uploaded_artifact"] = uploaded_data
            st.session_state["_uploaded_name"] = uploaded.name
        else:
            st.sidebar.error("Uploaded file does not look like an LNRA artifact.")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON file.")

# Build selection list (discovered + uploaded)
selection_options: list[str] = list(artifact_files)
if "_uploaded_name" in st.session_state:
    uname = f"[uploaded] {st.session_state['_uploaded_name']}"
    if uname not in selection_options:
        selection_options.insert(0, uname)

selected_artifact_key = st.sidebar.selectbox(
    "Select artifact",
    selection_options,
    index=0,
)

# Mode selector
MODES = ["Explore", "Query", "Compose", "Diff"]
mode = st.sidebar.radio("Mode", MODES, index=0)

if not HAS_API_KEY and mode in ("Query", "Compose", "Diff"):
    st.sidebar.warning(
        "ANTHROPIC_API_KEY is not set. Only **Explore** mode is available. "
        "Set the env var and restart to enable LLM features."
    )

st.sidebar.divider()
st.sidebar.markdown(
    "API Key: " + ("**Set**" if HAS_API_KEY else "**Not set** (Explore only)")
)


# ---------------------------------------------------------------------------
# Load selected artifact
# ---------------------------------------------------------------------------


def _get_artifact_raw(key: str) -> dict[str, Any]:
    if key.startswith("[uploaded]"):
        return st.session_state["_uploaded_artifact"]
    return load_artifact_json(key)


raw = _get_artifact_raw(selected_artifact_key)
meta = raw.get("metadata", {})

# ---------------------------------------------------------------------------
# Explore Mode
# ---------------------------------------------------------------------------


def render_explore(raw: dict[str, Any]):
    title = meta.get("title", "Untitled Artifact")
    st.header(title)

    # Metadata
    col1, col2, col3 = st.columns(3)
    col1.metric("Type", meta.get("artifact_type", "unknown"))
    col2.metric("Domain", meta.get("domain", "N/A"))
    col3.metric("Schema Version", meta.get("schema_version", "N/A"))

    tags = meta.get("tags", [])
    if tags:
        st.markdown("**Tags:** " + ", ".join(f"`{t}`" for t in tags))

    provenance = meta.get("provenance", [])
    if provenance:
        with st.expander("Provenance", expanded=False):
            for prov in provenance:
                st.markdown(
                    f"- **{prov.get('source_type', '')}** | "
                    f"{prov.get('source_id', '')} | "
                    f"confidence: {prov.get('confidence', 'N/A')}"
                )
                if prov.get("source_title"):
                    st.markdown(f"  *{prov['source_title']}*")

    st.divider()

    # Setup (experiment artifacts)
    setup = raw.get("setup")
    if setup:
        with st.expander("Experimental Setup", expanded=False):
            st.markdown(f"**Task:** {setup.get('task', 'N/A')}")
            st.markdown(f"**Methodology:** {setup.get('methodology', 'N/A')}")

            datasets = setup.get("datasets", [])
            if datasets:
                st.markdown("**Datasets:**")
                for ds in datasets:
                    st.markdown(
                        f"- {ds.get('name', 'unnamed')} "
                        f"(size: {ds.get('size', 'N/A')})"
                    )

            baselines = setup.get("baselines", [])
            if baselines:
                st.markdown("**Baselines:** " + ", ".join(baselines))

            hardware = setup.get("hardware")
            if hardware:
                st.markdown(f"**Hardware:** {hardware}")

            sw = setup.get("software_versions", {})
            if sw:
                st.markdown(
                    "**Software:** "
                    + ", ".join(f"{k}={v}" for k, v in sw.items())
                )

    # Claims
    claims = raw.get("claims", [])
    if claims:
        st.subheader(f"Claims ({len(claims)})")
        for i, claim in enumerate(claims):
            status = claim.get("status", "unknown")
            confidence = claim.get("confidence", "N/A")
            label = f"**Claim {i+1}** [{status}] (confidence: {confidence})"
            with st.expander(label, expanded=False):
                st.markdown(claim.get("statement", ""))

                ev_for = claim.get("evidence_for", [])
                if ev_for:
                    st.markdown("**Evidence for:**")
                    for ev in ev_for:
                        desc = ev.get("description", str(ev))
                        strength = ev.get("strength", "")
                        st.markdown(f"- {desc} (strength: {strength})")

                ev_against = claim.get("evidence_against", [])
                if ev_against:
                    st.markdown("**Evidence against:**")
                    for ev in ev_against:
                        desc = ev.get("description", str(ev))
                        st.markdown(f"- {desc}")

                conditions = claim.get("conditions", [])
                if conditions:
                    st.markdown("**Conditions:**")
                    for cond in conditions:
                        desc = cond.get("description", str(cond))
                        st.markdown(f"- [{cond.get('condition_type', '')}] {desc}")

    # Results as metrics table
    results = raw.get("results", [])
    if results:
        st.subheader("Results")
        _render_results_table(results)

    # Methods (method_comparison)
    methods = raw.get("methods", [])
    if methods:
        st.subheader(f"Methods ({len(methods)})")
        for method in methods:
            with st.expander(
                f"{method.get('name', 'unnamed')} [{method.get('category', '')}]",
                expanded=False,
            ):
                st.markdown(method.get("description", ""))
                if method.get("key_innovation"):
                    st.markdown(f"**Key Innovation:** {method['key_innovation']}")
                lims = method.get("limitations", [])
                if lims:
                    st.markdown("**Limitations:** " + "; ".join(lims))

    # Causal relationships
    causal = raw.get("causal_relationships", [])
    if causal:
        st.subheader(f"Causal Relationships ({len(causal)})")
        for rel in causal:
            with st.expander(
                f"{rel.get('cause', '?')} -> {rel.get('effect', '?')} "
                f"[{rel.get('strength', '')}]",
                expanded=False,
            ):
                if rel.get("mechanism"):
                    st.markdown(f"**Mechanism:** {rel['mechanism']}")
                confounders = rel.get("confounders", [])
                if confounders:
                    st.markdown("**Confounders:** " + ", ".join(confounders))

    # Failure cases
    failures = raw.get("failure_cases", [])
    if failures:
        st.subheader("Failure Cases")
        for fc in failures:
            st.markdown(f"- {fc}")

    # Key findings
    findings = raw.get("key_findings", [])
    if findings:
        st.subheader("Key Findings")
        for kf in findings:
            st.markdown(f"- {kf}")

    # Tradeoffs (method_comparison)
    tradeoffs = raw.get("tradeoffs", [])
    if tradeoffs:
        st.subheader("Tradeoff Analyses")
        for t in tradeoffs:
            with st.expander(t.get("description", "Tradeoff")[:80], expanded=False):
                st.markdown(t.get("description", ""))
                if t.get("recommendation"):
                    st.markdown(f"**Recommendation:** {t['recommendation']}")

    # Full JSON tree view
    st.divider()
    with st.expander("Full JSON", expanded=False):
        st.json(raw)


def _render_results_table(results: list[dict]):
    """Render results as a metrics table."""
    import pandas as pd

    # Collect all metric names
    metric_names: list[str] = []
    for r in results:
        for m in r.get("metrics", []):
            name = m.get("name", "")
            if name and name not in metric_names:
                metric_names.append(name)

    if not metric_names:
        st.info("No metrics found in results.")
        return

    rows = []
    for r in results:
        row: dict[str, Any] = {
            "Method": r.get("method_name", "unnamed"),
            "Proposed": r.get("is_proposed", False),
        }
        metrics_map = {m["name"]: m for m in r.get("metrics", []) if "name" in m}
        for mn in metric_names:
            if mn in metrics_map:
                val = metrics_map[mn].get("value", "")
                row[mn] = val
            else:
                row[mn] = None
        rows.append(row)

    df = pd.DataFrame(rows)

    # Highlight proposed methods
    def _highlight_proposed(row):
        if row.get("Proposed"):
            return ["background-color: #1a3a1a"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_proposed, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Query Mode
# ---------------------------------------------------------------------------


def render_query(raw: dict[str, Any]):
    st.header("Query Artifact")
    st.markdown(f"Querying: **{meta.get('title', 'Untitled')}**")

    if not HAS_API_KEY:
        st.warning(
            "ANTHROPIC_API_KEY is not set. "
            "Only programmatic queries (no LLM) are available."
        )

    query_type = st.radio(
        "Query type",
        ["Programmatic (no LLM)", "LLM-augmented"],
        horizontal=True,
        index=0,
    )

    question = st.text_area(
        "Ask a question about this artifact",
        placeholder=(
            "e.g., What are the key claims? "
            "Which method has the best accuracy? "
            "What are the preconditions?"
        ),
        height=100,
    )

    if st.button("Submit Query", type="primary", disabled=(not question)):
        artifact_obj = load_artifact_object(raw)

        if query_type == "Programmatic (no LLM)":
            from lnra.agent.interface import ArtifactAgent

            # We only use the programmatic path; no API key needed
            agent = ArtifactAgent.__new__(ArtifactAgent)
            result = agent._try_programmatic_query(artifact_obj, question)
            if result is None:
                st.info(
                    "No programmatic answer found for this question. "
                    "Try rephrasing or switch to LLM-augmented mode."
                )
                st.markdown(
                    "**Tip:** Programmatic queries work with keywords like: "
                    "*best*, *claim*, *finding*, *key*, *precondition*, *tradeoff*."
                )
            else:
                _render_query_result(result)
        else:
            if not HAS_API_KEY:
                st.error("ANTHROPIC_API_KEY is required for LLM-augmented queries.")
                return
            from lnra.agent.interface import ArtifactAgent

            agent = ArtifactAgent()
            with st.spinner("Querying with Claude..."):
                result = agent.query(artifact_obj, question)
            _render_query_result(result)


def _render_query_result(result: dict[str, Any]):
    """Render a structured query result."""
    st.subheader("Answer")
    st.markdown(result.get("answer", "No answer returned."))

    confidence = result.get("confidence")
    if confidence is not None:
        st.progress(float(confidence), text=f"Confidence: {confidence}")

    source = result.get("source")
    if source:
        st.caption(f"Source: {source}")

    evidence = result.get("evidence")
    if evidence:
        with st.expander("Evidence", expanded=True):
            if isinstance(evidence, list):
                for ev in evidence:
                    if isinstance(ev, dict):
                        st.json(ev)
                    else:
                        st.markdown(f"- {ev}")
            elif isinstance(evidence, dict):
                st.json(evidence)
            else:
                st.markdown(str(evidence))

    caveats = result.get("caveats", [])
    if caveats:
        with st.expander("Caveats"):
            for c in caveats:
                st.markdown(f"- {c}")

    conditions = result.get("relevant_conditions", [])
    if conditions:
        with st.expander("Relevant Conditions"):
            for c in conditions:
                st.markdown(f"- {c}")


# ---------------------------------------------------------------------------
# Compose Mode
# ---------------------------------------------------------------------------


def render_compose():
    st.header("Compose Artifacts")

    if not HAS_API_KEY:
        st.error(
            "ANTHROPIC_API_KEY is required for Compose mode. "
            "Set the env var and restart."
        )
        return

    st.markdown("Select two artifacts to compose and discover cross-artifact insights.")

    col1, col2 = st.columns(2)
    with col1:
        art_a_key = st.selectbox("Artifact A", selection_options, index=0, key="compose_a")
    with col2:
        idx_b = min(1, len(selection_options) - 1)
        art_b_key = st.selectbox("Artifact B", selection_options, index=idx_b, key="compose_b")

    question = st.text_input(
        "Guiding question (optional)",
        placeholder="e.g., What patterns emerge across these two studies?",
    )

    if st.button("Compose", type="primary"):
        raw_a = _get_artifact_raw(art_a_key)
        raw_b = _get_artifact_raw(art_b_key)
        obj_a = load_artifact_object(raw_a)
        obj_b = load_artifact_object(raw_b)

        from lnra.agent.interface import ArtifactAgent

        agent = ArtifactAgent()
        with st.spinner("Composing artifacts with Claude..."):
            result = agent.compose(
                [obj_a, obj_b], question=question if question else None
            )

        _render_compose_result(result)


def _render_compose_result(result: dict[str, Any]):
    """Render compose output."""
    st.subheader("Synthesis")
    st.markdown(result.get("synthesis", ""))

    shared = result.get("shared_findings", [])
    if shared:
        st.subheader("Shared Findings")
        for f in shared:
            st.markdown(f"- {f}")

    contradictions = result.get("contradictions", [])
    if contradictions:
        st.subheader("Contradictions")
        for c in contradictions:
            if isinstance(c, dict):
                st.error(
                    f"**{c.get('description', 'Contradiction')}**\n\n"
                    f"Possible explanation: {c.get('possible_explanation', 'N/A')}"
                )
            else:
                st.error(str(c))

    novel = result.get("novel_insights", [])
    if novel:
        st.subheader("Novel Insights")
        for n in novel:
            st.success(n)

    gaps = result.get("gaps", [])
    if gaps:
        st.subheader("Knowledge Gaps")
        for g in gaps:
            st.warning(g)

    conf = result.get("confidence")
    if conf is not None:
        st.progress(float(conf), text=f"Confidence: {conf}")

    rankings = result.get("method_rankings", {})
    if rankings:
        with st.expander("Method Rankings"):
            st.json(rankings)

    prog = result.get("programmatic_analysis")
    if prog:
        with st.expander("Programmatic Analysis (raw)"):
            st.json(prog)


# ---------------------------------------------------------------------------
# Diff Mode
# ---------------------------------------------------------------------------


def render_diff():
    st.header("Diff Artifacts")

    if not HAS_API_KEY:
        st.error(
            "ANTHROPIC_API_KEY is required for Diff mode. "
            "Set the env var and restart."
        )
        return

    st.markdown(
        "Compare two artifacts to find agreements, contradictions, "
        "and unique findings."
    )

    col1, col2 = st.columns(2)
    with col1:
        art_a_key = st.selectbox("Artifact A", selection_options, index=0, key="diff_a")
    with col2:
        idx_b = min(1, len(selection_options) - 1)
        art_b_key = st.selectbox("Artifact B", selection_options, index=idx_b, key="diff_b")

    if st.button("Run Diff", type="primary"):
        raw_a = _get_artifact_raw(art_a_key)
        raw_b = _get_artifact_raw(art_b_key)
        obj_a = load_artifact_object(raw_a)
        obj_b = load_artifact_object(raw_b)

        from lnra.agent.interface import ArtifactAgent

        agent = ArtifactAgent()
        with st.spinner("Diffing artifacts with Claude..."):
            result = agent.diff(obj_a, obj_b)

        _render_diff_result(result)


def _render_diff_result(result: dict[str, Any]):
    """Render diff output with color-coded sections."""
    st.subheader("Summary")
    st.markdown(result.get("summary", ""))

    # Agreements — green
    agreements = result.get("agreements", [])
    if agreements:
        st.subheader("Agreements")
        for a in agreements:
            if isinstance(a, dict):
                conf = a.get("confidence", "")
                conf_str = f" (confidence: {conf})" if conf else ""
                st.success(
                    f"**{a.get('topic', 'Agreement')}**{conf_str}\n\n"
                    f"{a.get('description', '')}"
                )
            else:
                st.success(str(a))

    # Contradictions — red
    contradictions = result.get("contradictions", [])
    if contradictions:
        st.subheader("Contradictions")
        for c in contradictions:
            if isinstance(c, dict):
                severity = c.get("severity", "")
                st.error(
                    f"**{c.get('topic', 'Contradiction')}** [{severity}]\n\n"
                    f"- Artifact A: {c.get('artifact_a_claim', 'N/A')}\n"
                    f"- Artifact B: {c.get('artifact_b_claim', 'N/A')}\n\n"
                    f"*Possible explanation:* {c.get('possible_explanation', 'N/A')}"
                )
            else:
                st.error(str(c))

    # Unique to A — blue
    unique_a = result.get("unique_to_a", [])
    if unique_a:
        st.subheader("Unique to Artifact A")
        for item in unique_a:
            st.info(str(item))

    # Unique to B — orange/warning
    unique_b = result.get("unique_to_b", [])
    if unique_b:
        st.subheader("Unique to Artifact B")
        for item in unique_b:
            st.warning(str(item))

    # Complementary findings
    complementary = result.get("complementary", [])
    if complementary:
        st.subheader("Complementary Findings")
        for item in complementary:
            st.markdown(f"- {item}")

    # Methodology differences
    method_diff = result.get("methodology_differences", [])
    if method_diff:
        with st.expander("Methodology Differences"):
            for item in method_diff:
                st.markdown(f"- {item}")

    # Recommendation
    rec = result.get("recommendation")
    if rec:
        st.subheader("Recommendation")
        st.markdown(rec)

    # Programmatic diff
    prog = result.get("programmatic_diff")
    if prog:
        with st.expander("Programmatic Diff (raw)"):
            st.json(prog)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

if mode == "Explore":
    render_explore(raw)
elif mode == "Query":
    render_query(raw)
elif mode == "Compose":
    render_compose()
elif mode == "Diff":
    render_diff()
