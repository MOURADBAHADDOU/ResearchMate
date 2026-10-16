##answer.py
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from langdetect import detect, LangDetectException


# ============================================================
# 1. Configuration du projet
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "research_mate"
LLM_MODEL = "qwen2.5:3b"

TOP_K = 5

# Langue autorisée pour les questions
ALLOWED_LANGUAGE = "en"

# Message de refus si la question n'est pas en anglais
NON_ENGLISH_MESSAGE = (
    "Research Mate only accepts questions in English. "
    "Please rephrase your question in English.\n\n"
    "Example: 'What is weight decay?' instead of "
    "'Qu'est-ce que le weight decay ?'."
)


# ============================================================
# 2. Instructions données au LLM
# ============================================================

SYSTEM_PROMPT = """
You are Research Mate, an educational assistant specialized in
Statistics, Machine Learning, and Deep Learning.

Your task is to answer students' questions using the course
excerpts provided in the context.

FUNDAMENTAL RULES

1. STRICT USE OF CONTEXT
- You ONLY answer questions written in English.
- If the question is written in any other language (French,
  Spanish, German, Arabic, etc.), you MUST refuse to answer.
- In that case, respond exactly with:
  "Research Mate only accepts questions in English.
   Please rephrase your question in English."
- Do NOT translate the question. Do NOT answer it.
- Do NOT provide any other explanation, definition, or comment.
- Use the retrieved excerpts as the evidence for factual claims.
- Do not present general knowledge as information found in
  the provided documents.
- Do not introduce definitions, formulas, methods, or claims
  that are not supported by the context.
- You may reformulate and explain the excerpts without
  changing their meaning.
- If an illustrative example is not taken from the documents,
  explicitly identify it as an illustrative example.

2. INSUFFICIENT CONTEXT
- If the excerpts only partially answer the question, answer
  only the parts supported by the available evidence.
- Explicitly identify information that is missing.
- Never pretend to have checked the entire PDF when only
  a few excerpts were retrieved.
- If no excerpt answers the question reliably, say so.

3. ACCURACY
- Preserve the assumptions, conditions, units, notation,
  equations, and limitations given in the sources.
- Do not turn correlation into causation.
- Do not present a possibility as a certainty.
- If a question contains an incorrect premise, politely
  point it out.
- For mathematical questions, explain the necessary steps
  and remain consistent with the available evidence.

4. RELEVANCE
- Answer the exact question.
- Organize the explanation logically.
- Avoid repetition, generic introductions, and irrelevant
  information.
- If several excerpts express the same idea, synthesize them.
- Answer every sub-question when the request has several parts.

5. ADAPT TO THE QUESTION
- For a definition, give a clear and precise definition.
- For a comparison, explain relevant similarities and differences.
- For a formula, explain its symbols and necessary steps.
- For an algorithm, explain its principle and limitations
  when the sources provide that information.
- For a broad question, use short, meaningful sections.
- Adapt the length to the complexity of the question.

6. CITATIONS AND TRACEABILITY
- Cite the source near each important factual claim using
  the exact filename and page number provided in the context.
- Use only source references actually provided.
- Never invent filenames, page numbers, or citations.
- Each citation must genuinely support the associated claim.
- If no excerpt supports a claim, omit it or explicitly state
  that the available context does not support it.
- Do not assume that a retrieved excerpt is relevant merely
  because it was returned by the search engine.

7. METHODS AND RECOMMENDATIONS
- Before recommending a technique, check whether the context
  actually describes or supports it.
- Do not list familiar Machine Learning methods simply
  because they are generally known.
- If the sources mention a method without explaining it,
  do not invent an explanation attributed to those sources.
- Clearly state when the available evidence is incomplete.

8. LANGUAGE AND QUALITY
- Always answer in English. Never answer in another language.
- Use clear, natural, pedagogical English.
- Define technical terms necessary for understanding.
- Avoid vague wording and incomplete sentences.

Choose the structure that best fits the question.
When appropriate, include:
- Direct answer
- Explanation or necessary steps
- Formula or illustrative example, if useful
- Inline citations such as [ML.pdf, p. 31]

Do not add empty sections or repeat all sources unnecessarily.



MATHEMATICAL FORMATTING

- Use $...$ for inline mathematical expressions.
- Use $$...$$ on separate lines for displayed equations.
- Never put mathematical expressions in plain parentheses
  when proper mathematical notation is needed.
- Never escape underscores unnecessarily. Write x_i, not x\_i,
  inside LaTeX expressions.
- Example of inline math: The sample mean is $\bar{x}$.
- Example of a displayed equation:

$$
z_m = \frac{1}{m}\sum_{i=1}^{m}x_i
$$

- Keep all LaTeX delimiters balanced.
- Do not put mathematical formulas inside Markdown code blocks.


IMPORTANT
The retrieved context may be incomplete or irrelevant.
A retrieved excerpt does not automatically constitute evidence.
Prefer a partial, honest answer over a complete answer
containing unsupported claims.
Answer solely based on the provided passages. For each statement,
cite the exact passage. If information is not found in the passages,
state this explicitly.
"""


# ============================================================
# 3. Fonctions utilitaires pour la détection de langue
# ============================================================

def is_english(text: str) -> bool:
    """Détecte si le texte est en anglais."""
    try:
        return detect(text) == ALLOWED_LANGUAGE
    except LangDetectException:
        # En cas d'échec (texte trop court, symboles, etc.),
        # on laisse passer pour ne pas bloquer l'utilisateur.
        return True


def has_non_english_chars(text: str) -> bool:
    """Détecte des caractères typiques du français/espagnol/etc."""
    return bool(
        re.search(
            r"[àâäéèêëîïôöùûüçñáíóúü¿¡]",
            text,
            re.IGNORECASE,
        )
    )


# ============================================================
# 4. Chargement des composants RAG
# ============================================================

def load_rag_components():
    """Charge les composants nécessaires au RAG."""

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Base vectorielle introuvable : {DB_DIR}. "
            "Exécute build_index.py."
        )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' introuvable."
        ) from exc

    if collection.count() == 0:
        raise ValueError("La collection ChromaDB est vide.")

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    return embedding_model, collection, llm


# ============================================================
# 5. Fonction appelée par Streamlit
# ============================================================

def answer_question(
    question: str,
    embedding_model,
    collection,
    llm,
) -> dict:
    """Répond à une question et retourne les sources utilisées."""

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    if has_non_english_chars(question) or not is_english(question):
        return {
            "answer": NON_ENGLISH_MESSAGE,
            "sources": [],
        }

    query = (
        "Represent this sentence for searching relevant passages: "
        + question
    )

    question_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    number_of_results = min(TOP_K, collection.count())

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    # Construire les sources avec leurs références.
    sources = []
    context_parts = []

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        metadata = metadata or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        sources.append({
            "rank": rank,
            "source": source,
            "page": page,
            "distance": float(distance),
            "text": document or "",
        })

        context_parts.append(
            f"[Passage {rank} | Source: {source} | Page: {page}]\n"
            f"{document}"
        )

    if not context_parts:
        return {
            "answer": "No relevant passages were retrieved.",
            "sources": [],
        }

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Retrieved course excerpts:\n{context}\n\n"
                "Answer the question using the rules provided. "
                "Cite the relevant sources near the claims they support."
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources,
    }


# ============================================================
# 6. Utilisation depuis le terminal (facultatif)
# ============================================================

def main() -> None:
    print("Loading Research Mate...")

    embedding_model, collection, llm = load_rag_components()

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        result = answer_question(
            question,
            embedding_model,
            collection,
            llm,
        )

        print("\n========== ANSWER ==========\n")
        print(result["answer"])

        for source in result["sources"]:
            print(
                f"\nSource: {source['source']} | "
                f"Page: {source['page']} | "
                f"Distance: {source['distance']:.4f}"
            )


if __name__ == "__main__":
    main()
