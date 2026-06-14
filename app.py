import streamlit as st

from generation.analyzer import analyze_job_with_evidence
from rag.retriever import (
    format_evidence_for_prompt,
    ingest_profile,
    retrieve_relevant_evidence,
)

st.set_page_config(
    page_title="RAG CV Builder",
    page_icon="📄",
    layout="wide",
)

st.title("RAG CV Builder for Navneeth Rajesh")
st.caption(
    "Analyze a job description, retrieve relevant candidate evidence, "
    "and prepare a tailored CV."
)

with st.sidebar:
    st.header("Setup")

    if st.button("Ingest / Refresh Candidate Profile"):
        with st.spinner("Building candidate vector store using LM Studio embeddings..."):
            count = ingest_profile(reset=True)
        st.success(f"Ingested {count} candidate evidence items.")

    top_k = st.slider("Evidence items to retrieve", 3, 12, 8)

    st.markdown("---")
    st.markdown("### Current version")
    st.write("v0.1: Job analysis and evidence retrieval")
    st.write("v0.2: Tailored CV generation")

job_description = st.text_area(
    "Paste the job description here",
    height=350,
    placeholder="Paste the full job description...",
)

if st.button("Analyze Role", type="primary"):
    if not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        with st.spinner("Retrieving relevant candidate evidence..."):
            evidence = retrieve_relevant_evidence(job_description, top_k=top_k)
            evidence_text = format_evidence_for_prompt(evidence)

        st.subheader("Retrieved Evidence")

        for idx, item in enumerate(evidence, start=1):
            metadata = item["metadata"]

            with st.expander(
                f"{idx}. {metadata.get('title')} — {metadata.get('type')}"
            ):
                st.write(item["text"])
                st.caption(
                    f"Source ID: {metadata.get('source_id')} | "
                    f"Distance: {item.get('distance')}"
                )

        with st.spinner("Analyzing fit using LM Studio..."):
            analysis = analyze_job_with_evidence(job_description, evidence_text)

        st.subheader("Role Fit Analysis")
        st.markdown(analysis)

        st.markdown("---")
        st.info("Next version: add evidence selection and one-page CV generation.")