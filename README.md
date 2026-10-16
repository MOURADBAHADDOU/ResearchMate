# Research Mate — RAG-Based Learning Assistant

Research Mate is a Retrieval-Augmented Generation (RAG) application designed to help students understand concepts in **Statistics, Machine Learning, and Deep Learning**.

The application retrieves relevant passages from educational PDF documents and uses a locally running Large Language Model (LLM) to generate context-grounded answers with source citations.

## Features

- **PDF document ingestion:** Extracts text from Statistics, Machine Learning, and Deep Learning textbooks.
- **Text chunking:** Splits documents into smaller, overlapping text segments.
- **Semantic search:** Retrieves relevant passages using vector embeddings.
- **Vector database:** Stores and searches document embeddings with ChromaDB.
- **Context-grounded answers:** Generates answers using retrieved passages and a local LLM.
- **Source citations:** Includes document names and page numbers to help users verify answers.
- **Interactive interface:** Provides a user interface built with Streamlit.
- **Local LLM inference:** Uses Ollama to run the language model locally.

## Technologies

- **Programming language:** Python
- **RAG:** Retrieval-Augmented Generation
- **Embeddings:** Sentence Transformers — `BAAI/bge-small-en-v1.5`
- **Vector database:** ChromaDB
- **LLM integration:** LangChain and Ollama
- **Language model:** `qwen2.5:3b`
- **PDF processing:** PyMuPDF
- **Interface:** Streamlit

## Project Structure

```text
ResearchMate/
├── app.py
├── requirements.txt
├── README.md
├── pyproject.toml
├── src/
│   └── researchmate/
│       ├── ingest.py
│       ├── build_index.py
│       ├── search.py
│       ├── answer.py
│       └── renderer.py
├── tests/
│   ├── evaluate_retriever.py
│   ├── inspect_chunks.py
│   └── retriever_eval.json
└── data/
    ├── Statistics.pdf
    ├── ML.pdf
    ├── DL.pdf
    ├── processed/
    └── chroma_db/
```

*Note: This structure illustrates the main project files. Generated data and local files may be excluded from the Git repository.*

## Prerequisites

Before running Research Mate, install:

- Python 3.13 or a compatible version supported by the dependencies.
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Git

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ResearchMate.git
cd ResearchMate
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Python dependencies

```bash
uv pip install -r requirements.txt
```

If you are using the project's environment managed by uv, you can alternatively synchronize the project dependencies with:

```bash
uv sync
```

### 3. Download the language model

Make sure Ollama is installed and running, then execute:

```bash
ollama pull qwen2.5:3b
```

### 4. Prepare the documents

Place the educational PDF files in the `data/` directory, following the paths expected by the ingestion script.

### 5. Ingest the documents

```bash
uv run python src/researchmate/ingest.py
```

This extracts the PDF text and generates the processed text chunks.

### 6. Build the vector index

```bash
uv run python src/researchmate/build_index.py
```

This creates the ChromaDB vector index used for semantic retrieval.

### 7. Launch the application

```bash
uv run streamlit run app.py
```

Open the local URL displayed in your terminal to interact with Research Mate.

## How It Works

1. **Document ingestion:** Text is extracted from educational PDF files.
2. **Chunking:** The extracted text is divided into smaller overlapping segments.
3. **Embedding:** Each segment is converted into a numerical vector using a sentence-transformer model.
4. **Indexing:** The vectors and associated metadata are stored in ChromaDB.
5. **Retrieval:** A user's question is embedded and compared with the indexed passages.
6. **Generation:** The retrieved context is passed to the local LLM to generate an answer.
7. **Citation:** The answer includes document names and page references when supported by the retrieved context.

## Evaluation

The project includes scripts for inspecting retrieved passages and evaluating retrieval quality.

```bash
uv run python tests/inspect_chunks.py
uv run python tests/evaluate_retriever.py
```

The evaluation dataset is stored in `tests/retriever_eval.json`.

Retrieval performance should be assessed using appropriate metrics, such as Precision@K, Recall@K, and Mean Reciprocal Rank (MRR), provided these metrics are implemented in the evaluation script.

## Limitations

- Answer quality depends on the relevance of the retrieved passages and the capabilities of the selected language model.
- The application is limited to the documents that have been indexed.
- Local inference performance depends on available hardware and model size.
- Generated answers should be checked against the cited source material.

## Future Improvements

- Improve retrieval quality through systematic evaluation and parameter tuning.
- Compare embedding models and retrieval strategies.
- Add reranking to improve the relevance of retrieved passages.
- Expand the evaluation dataset and report quantitative results.
- Improve answer faithfulness and citation verification.

## Author

**YOUR NAME**

Data Science Engineering Student

[GitHub Profile](https://github.com/YOUR_USERNAME)

---

*Research Mate is an educational project exploring Retrieval-Augmented Generation, semantic search, vector databases, and locally hosted Large Language Models.*