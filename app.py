import os
from pathlib import Path

# Build vector store on first run if missing (for Streamlit Cloud)
if not Path("./chroma_db").exists():
    from src.vectorstore import create_vector_store
    create_vector_store()


import streamlit as st

from src.rag import RAGPipeline


# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Northbridge Policy Assistant",
    page_icon="📚",
    layout="centered",
)


# --------------------------------------------------
# RAG PIPELINE
# --------------------------------------------------

@st.cache_resource
def load_rag():
    return RAGPipeline()


rag = load_rag()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📚 Northbridge Policy Assistant")

st.caption(
    "Ask questions about the university policies provided to the system."
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                st.markdown("### Sources")

                displayed = set()

                for source in sources:

                    key = (
                        source["source"],
                        source["page"],
                        source["section"],
                    )

                    if key in displayed:
                        continue

                    displayed.add(key)

                    st.markdown(
                        f"**📄 {source['source']}**  \n"
                        f"Page {source['page']} · "
                        f"{source['section']}"
                    )

                # Optional debugging information
                with st.expander("View retrieved evidence"):

                    for source in sources:

                        st.markdown(
                            f"**{source['source']}** · "
                            f"Page {source['page']} · "
                            f"{source['section']}"
                        )

                        st.caption(source["text"])


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about the university policies..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching the policy documents..."):

            result = rag.answer(question)

        st.markdown(result["answer"])

        sources = result.get("sources", [])

        if sources:

            st.markdown("### Sources")

            displayed = set()

            for source in sources:

                key = (
                    source["source"],
                    source["page"],
                    source["section"],
                )

                if key in displayed:
                    continue

                displayed.add(key)

                st.markdown(
                    f"**📄 {source['source']}**  \n"
                    f"Page {source['page']} · "
                    f"{source['section']}"
                )

            with st.expander("View retrieved evidence"):

                for source in sources:

                    st.markdown(
                        f"**{source['source']}** · "
                        f"Page {source['page']} · "
                        f"{source['section']}"
                    )

                    st.caption(source["text"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": sources,
            }
        )
