
from pathlib import Path
import sys

import streamlit as st

# Permet d'importer le package situé dans src/
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchmate.answer import (
    answer_question,
    load_rag_components,
)

# MODIFICATION 1 : importer le renderer mathématique
from researchmate.renderer import render_answer


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Research Mate",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Mate")
st.caption(
    "Your AI study assistant for Statistics, "
    "Machine Learning and Deep Learning."
)


# ============================================================
# Charger les composants une seule fois dans le cache
# ============================================================

@st.cache_resource(show_spinner="Loading the RAG engine...")
def get_rag_components():
    return load_rag_components()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("About Research Mate")

    st.write(
        "Ask questions about your course materials "
        "and explore the retrieved PDF passages."
    )

    st.divider()

    st.subheader("Knowledge base")
    st.write("📊 Statistics.pdf")
    st.write("🤖 ML.pdf")
    st.write("🧠 DL.pdf")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by ChromaDB, BGE and Ollama.")


# ============================================================
# Initialiser l'historique
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Charger les composants RAG
# ============================================================

try:
    embedding_model, collection, llm = get_rag_components()
except Exception as exc:
    st.error(
        "Unable to initialize Research Mate. "
        "Check the ChromaDB index and Ollama installation."
    )
    st.exception(exc)
    st.stop()


# ============================================================
# Afficher l'historique
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):

        # MODIFICATION 2 : utiliser le renderer pour les réponses
        # de l'assistant, et conserver Markdown pour l'utilisateur
        if message["role"] == "assistant":
            render_answer(message["content"])
        else:
            st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander(
                    f"📄 Retrieved sources ({len(sources)})"
                ):
                    for source in sources:
                        st.markdown(
                            f"**{source['source']} — "
                            f"page {source['page']}**"
                        )

                        st.caption(
                            f"Retrieval distance: "
                            f"{source['distance']:.4f}"
                        )

                        st.write(
                            source["text"][:1200]
                            + (
                                "..."
                                if len(source["text"]) > 1200
                                else ""
                            )
                        )

                        st.divider()


# ============================================================
# Saisie et traitement de la question
# ============================================================

question = st.chat_input(
    "Ask a question about Statistics, ML or DL..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching the courses and generating an answer..."
        ):
            try:
                result = answer_question(
                    question=question,
                    embedding_model=embedding_model,
                    collection=collection,
                    llm=llm,
                )

                response_text = result["answer"]
                sources = result["sources"]

                # MODIFICATION 3 : rendre les formules LaTeX
                # au lieu d'afficher toute la réponse en Markdown brut
                render_answer(response_text)

                if sources:
                    with st.expander(
                        f"📄 Retrieved sources ({len(sources)})",
                        expanded=False,
                    ):
                        for source in sources:
                            st.markdown(
                                f"**{source['source']} — "
                                f"page {source['page']}**"
                            )

                            st.caption(
                                f"Retrieval distance: "
                                f"{source['distance']:.4f}"
                            )

                            st.write(
                                source["text"][:1200]
                                + (
                                    "..."
                                    if len(source["text"]) > 1200
                                    else ""
                                )
                            )

                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources,
                })

            except Exception as exc:
                st.error(
                    "An error occurred while generating the answer."
                )
                st.exception(exc)
