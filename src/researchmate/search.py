
##search.py
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "research_mate"

# Seuil initial à ajuster après évaluation
MAX_DISTANCE = 0.40
TOP_K = 5

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


def main() -> None:
    print("Chargement du modèle d'embeddings...")
    model = SentenceTransformer(MODEL_NAME)

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Base vectorielle introuvable : {DB_DIR}\n"
            "Exécute d'abord build_index.py."
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' introuvable. "
            "Exécute build_index.py pour construire l'index."
        ) from exc

    if collection.count() == 0:
        print("La base vectorielle ne contient aucun document.")
        return

    question = input("\nPose ta question : ").strip()

    if not question:
        print("La question ne peut pas être vide.")
        return

    # Préparer la requête pour le modèle BGE
    query = QUERY_PREFIX + question

    question_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    print("\n--- Résultats de la recherche ---")
    relevant_count = 0

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"\nRésultat {rank}")
        print(f"Source : {metadata.get('source', 'Inconnue')}")
        print(f"Page : {metadata.get('page', 'Inconnue')}")
        print(f"Distance : {distance:.4f}")

        if distance > MAX_DISTANCE:
            print("Passage ignoré : pertinence insuffisante.")
            continue

        relevant_count += 1
        print(f"Extrait :\n{document[:700]}")
        print("-" * 60)

    if relevant_count == 0:
        print(
            "\nAucun passage ne satisfait le seuil de pertinence."
            "\nEssaie de reformuler la question."
        )


if __name__ == "__main__":
    main()
