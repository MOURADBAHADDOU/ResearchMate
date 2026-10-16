##build_index.py
from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Même modèle que dans search.py et answer.py
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "research_mate"


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CHUNKS_FILE}\n"
            "Exécute d'abord ingest.py."
        )

    # Charger les chunks
    with CHUNKS_FILE.open(encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not chunks:
        raise ValueError(
            "Aucun chunk trouvé dans chunks.jsonl."
        )

    print(f"Chunks chargés : {len(chunks)}")

    # Charger le modèle d'embeddings
    print(f"Chargement du modèle : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Créer la base vectorielle persistante
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Supprimer l'ancienne collection pour reconstruire l'index
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Ancienne collection supprimée.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Ajouter les chunks par lots
    batch_size = 64

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]
        ids = [chunk["chunk_id"] for chunk in batch]

        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
            }
            for chunk in batch
        ]

        # Encoder les documents sans préfixe de requête
        embeddings = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        print(
            f"Indexation : "
            f"{min(start + len(batch), len(chunks))}"
            f" / {len(chunks)}"
        )

    print("\nIndexation terminée !")
    print(f"Nombre de vecteurs : {collection.count()}")
    print(f"Base enregistrée dans : {DB_DIR}")


if __name__ == "__main__":
    main()
