
import csv
import json
import math
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
EVAL_PATH = PROJECT_ROOT / "tests" / "retriever_eval.json"
OUTPUT_PATH = PROJECT_ROOT / "tests" / "retriever_results.csv"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "research_mate"

K_VALUES = [3, 5, 10]

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


# ============================================================
# 2. MÉTRIQUES D'ÉVALUATION
# ============================================================

def precision_at_k(ranked_ids, relevant_ids, k):
    """Proportion de chunks pertinents parmi les K premiers."""
    if k <= 0:
        raise ValueError("K doit être strictement positif.")

    top_k = ranked_ids[:k]
    hits = sum(chunk_id in relevant_ids for chunk_id in top_k)

    # On divise par le nombre de résultats réellement évalués.
    # Ici, on récupère normalement au moins K résultats.
    denominator = len(top_k)

    return hits / denominator if denominator else 0.0


def recall_at_k(ranked_ids, relevant_ids, k):
    """Proportion des chunks pertinents annotés retrouvés dans le top K."""
    if k <= 0:
        raise ValueError("K doit être strictement positif.")

    if not relevant_ids:
        return 0.0

    top_k = ranked_ids[:k]
    hits = sum(chunk_id in relevant_ids for chunk_id in top_k)

    return hits / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    """Réciproque du rang du premier chunk pertinent."""
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    """Qualité du classement, avec plus de poids pour les premiers rangs."""
    if k <= 0:
        raise ValueError("K doit être strictement positif.")

    top_k = ranked_ids[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top_k, start=1)
        if chunk_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(ranked_ids, relevant_ids, k):
    """Vaut 1 si au moins un chunk pertinent apparaît dans le top K."""
    if k <= 0:
        raise ValueError("K doit être strictement positif.")

    return float(
        any(chunk_id in relevant_ids for chunk_id in ranked_ids[:k])
    )


# ============================================================
# 3. CHARGEMENT ET VALIDATION DU JEU DE TEST
# ============================================================

def load_evaluation_data():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Fichier d'évaluation introuvable : {EVAL_PATH}"
        )

    with open(EVAL_PATH, "r", encoding="utf-8") as file:
        evaluation_data = json.load(file)

    if not isinstance(evaluation_data, list) or not evaluation_data:
        raise ValueError(
            "Le jeu d'évaluation doit être une liste non vide de questions."
        )

    seen_questions = set()

    for index, item in enumerate(evaluation_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Entrée numéro {index} invalide.")

        question = item.get("question")
        relevant_ids = item.get("relevant_chunk_ids")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Question absente ou vide à l'entrée {index}."
            )

        if question in seen_questions:
            raise ValueError(f"Question dupliquée : {question}")

        seen_questions.add(question)

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(
                f"Aucun chunk pertinent annoté pour : {question}"
            )

        if not all(
            isinstance(chunk_id, str) and chunk_id.strip()
            for chunk_id in relevant_ids
        ):
            raise ValueError(
                f"Identifiant de chunk invalide pour : {question}"
            )

        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(
                f"Identifiants dupliqués pour : {question}"
            )

    return evaluation_data


# ============================================================
# 4. ÉVALUATION DU RETRIEVER
# ============================================================

def main():
    evaluation_data = load_evaluation_data()

    if not K_VALUES or any(k <= 0 for k in K_VALUES):
        raise ValueError("K_VALUES doit contenir des entiers positifs.")

    if len(K_VALUES) != len(set(K_VALUES)):
        raise ValueError("K_VALUES ne doit pas contenir de doublons.")

    max_k = max(K_VALUES)

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

    # Vérifie que les chunks annotés existent dans la collection.
    all_relevant_ids = {
        chunk_id
        for item in evaluation_data
        for chunk_id in item["relevant_chunk_ids"]
    }

    existing_ids = set()

    relevant_id_list = list(all_relevant_ids)

    # Lecture par lots pour éviter une requête trop volumineuse.
    batch_size = 500

    for start in range(0, len(relevant_id_list), batch_size):
        batch = relevant_id_list[start:start + batch_size]

        existing = collection.get(
            ids=batch,
            include=["metadatas"],
        )

        existing_ids.update(existing["ids"])

    missing_ids = all_relevant_ids - existing_ids

    if missing_ids:
        raise ValueError(
            "Certains IDs annotés sont absents de ChromaDB : "
            f"{sorted(missing_ids)}"
        )

    print(f"Collection : {collection.name}")
    print(f"Nombre de chunks : {collection_size}")
    print(f"Nombre de questions : {len(evaluation_data)}")
    print(f"Valeurs de K : {K_VALUES}")

    rows = []

    for item in evaluation_data:
        question = item["question"]
        relevant_ids = set(item["relevant_chunk_ids"])

        query_embedding = model.encode(
            [QUERY_PREFIX + question],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(max_k, collection_size),
            include=["distances"],
        )

        ranked_ids = results["ids"][0]
        distances = results["distances"][0]

        print(f"\nQuestion : {question}")
        print("Classement :")

        for rank, (chunk_id, distance) in enumerate(
            zip(ranked_ids, distances), start=1
        ):
            marker = (
                "PERTINENT"
                if chunk_id in relevant_ids
                else "non annoté"
            )

            print(
                f"  {rank}. {chunk_id} | "
                f"distance={distance:.4f} | {marker}"
            )

        for k in K_VALUES:
            # Le MRR@K doit être calculé uniquement sur les K premiers.
            top_k_ids = ranked_ids[:k]

            row = {
                "question": question,
                "k": k,
                "precision_at_k": precision_at_k(
                    ranked_ids, relevant_ids, k
                ),
                "recall_at_k": recall_at_k(
                    ranked_ids, relevant_ids, k
                ),
                "mrr": reciprocal_rank(
                    top_k_ids, relevant_ids
                ),
                "ndcg_at_k": ndcg_at_k(
                    ranked_ids, relevant_ids, k
                ),
                "hit_rate_at_k": hit_rate_at_k(
                    ranked_ids, relevant_ids, k
                ),
            }

            rows.append(row)

    # ========================================================
    # 5. CALCUL DES MOYENNES
    # ========================================================

    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "hit_rate_at_k",
    ]

    summary = []

    print("\n" + "=" * 75)
    print("RÉSULTATS MOYENS")
    print("=" * 75)

    for k in K_VALUES:
        subset = [row for row in rows if row["k"] == k]

        if not subset:
            continue

        averages = {
            metric: sum(row[metric] for row in subset) / len(subset)
            for metric in metric_names
        }

        summary.append({"k": k, **averages})

        print(f"\nK = {k}")

        for metric, value in averages.items():
            print(f"  {metric:20s}: {value:.4f}")

    # ========================================================
    # 6. EXPORT CSV
    # ========================================================

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        fieldnames = ["question", "k", *metric_names]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRésultats détaillés enregistrés dans : {OUTPUT_PATH}")

    print("\nRappel : les résultats dépendent de la qualité des annotations.")
    print(
        "Un chunk non annoté n'est pas nécessairement non pertinent."
    )


if __name__ == "__main__":
    main()

