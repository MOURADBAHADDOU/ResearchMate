

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "research_mate"
TOP_K = 5
EXCERPT_LENGTH = 500

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


# ============================================================
# 2. JEU DE TEST : 20 QUESTIONS
# ============================================================

questions = [
    # STATISTICS (6 questions)
    "What is the difference between a qualitative and a quantitative variable?",

    "Explain the difference between the mean, the median, and the mode. "
    "When should I use each one?",

    "What is a confidence interval, and how should I interpret it correctly?",

    "What is a p-value, and why is it often misinterpreted?",

    "What is the difference between a Type I error and a Type II error?",

    "What is correlation, and why does correlation not imply causation?",

    # MACHINE LEARNING (6 questions)
    "What is the difference between supervised and unsupervised learning?",

    "Explain the bias-variance trade-off. Why does test error have a U-shape?",

    "What is overfitting, and how can we prevent it?",

    "What is the difference between ridge regression and the lasso? "
    "When would I prefer one over the other?",

    "Explain how k-fold cross-validation works and why it is preferred "
    "over the validation set approach.",

    "What is the difference between bagging, random forests, and boosting?",

    # DEEP LEARNING (8 questions)
    "What is a neural network, and how does it differ from a linear "
    "regression model?",

    "Why do we need a non-linear activation function like ReLU or sigmoid?",

    "What is backpropagation, and how does it work?",

    "What is the difference between a CNN and a regular fully-connected "
    "network? Why are CNNs better for images?",

    "What is the vanishing gradient problem, and how do LSTMs or GRUs "
    "help solve it?",

    "Explain the attention mechanism and why it is useful for "
    "sequence-to-sequence tasks.",

    "What is the difference between a Transformer and an RNN? "
    "Why have Transformers become dominant?",

    "What is linear regression?",
]


# ============================================================
# 3. CHARGEMENT DU MODÈLE ET DE CHROMADB
# ============================================================

def main():
    print("Chargement du modèle d'embeddings...")

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_PATH))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de charger la collection '{COLLECTION_NAME}'. "
            f"Vérifie le chemin : {DB_PATH}"
        ) from exc

    collection_size = collection.count()

    if collection_size == 0:
        raise ValueError("La collection ChromaDB est vide.")

    print(f"\nCollection : {collection.name}")
    print(f"Nombre de chunks : {collection_size}")
    print(f"Nombre de questions : {len(questions)}")
    print(f"TOP_K : {TOP_K}")

    # ========================================================
    # 4. RECHERCHE POUR CHAQUE QUESTION
    # ========================================================

    for question_number, question in enumerate(questions, start=1):

        print("\n" + "=" * 90)
        print(f"QUESTION {question_number}/{len(questions)}")
        print(question)
        print("=" * 90)

        # Utilise le même modèle et le même préfixe que le retriever.
        query_embedding = model.encode(
            [QUERY_PREFIX + question],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(TOP_K, collection_size),
            include=["documents", "metadatas", "distances"],
        )

        result_ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(result_ids, documents, metadatas, distances),
            start=1,
        ):
            metadata = metadata or {}
            document = document or ""

            source = metadata.get("source", "inconnue")
            page = metadata.get("page", "inconnue")

            print(f"\n--- Résultat {rank} ---")
            print(f"ID       : {chunk_id}")
            print(f"Source   : {source}")
            print(f"Page     : {page}")
            print(f"Distance : {distance:.4f}")
            print(f"Extrait  : {document[:EXCERPT_LENGTH]}")

            if len(document) > EXCERPT_LENGTH:
                print("           [...]")

    print("\n" + "=" * 90)
    print("INSPECTION TERMINÉE")
    print(f"Questions inspectées : {len(questions)}")
    print(f"Résultats affichés au maximum : {len(questions) * TOP_K}")
    print("=" * 90)


if __name__ == "__main__":
    main()

