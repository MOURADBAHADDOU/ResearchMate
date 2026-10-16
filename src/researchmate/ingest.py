##ingest.py
from pathlib import Path
import json

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

PDF_FILES = [
    "Statistics.pdf",
    "ML.pdf",
    "DL.pdf",
]

# Configuration du découpage
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def process_pdf(pdf_path: Path) -> list[dict]:
    """Extrait le texte d'un PDF et le découpe en chunks."""
    chunks = []

    print(f"\nTraitement de : {pdf_path.name}")

    with pymupdf.open(pdf_path) as document:
        print(f"Nombre de pages : {len(document)}")

        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            # Ignorer les pages sans texte exploitable
            if not text:
                continue

            page_chunks = text_splitter.split_text(text)

            for chunk_index, chunk_text in enumerate(page_chunks):
                chunks.append({
                    "chunk_id": (
                        f"{pdf_path.stem}_p{page_index + 1}"
                        f"_c{chunk_index + 1}"
                    ),
                    "source": pdf_path.name,
                    "page": page_index + 1,
                    "text": chunk_text,
                })

    print(f"Chunks créés : {len(chunks)}")
    return chunks


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for filename in PDF_FILES:
        pdf_path = DATA_DIR / filename

        if not pdf_path.exists():
            print(f"Fichier introuvable : {pdf_path}")
            continue

        all_chunks.extend(process_pdf(pdf_path))

    # Enregistrer un chunk JSON par ligne
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\nExtraction terminée !")
    print(f"Nombre total de chunks : {len(all_chunks)}")
    print(f"Fichier généré : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

