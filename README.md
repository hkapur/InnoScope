# 🔭 InnoScope

**Retrieval-Augmented Decision Intelligence for Software R\&D**

InnoScope is a Retrieval-Augmented Generation (RAG) Proof of Concept (POC) designed to assist R\&D teams in making technical decisions. It ingests technical artifacts (PDFs), indexes them using a hybrid search strategy with re-ranking, and uses a multi-model architecture to generate and audit technical briefs.

## 🚀 Key Features

  * **Hybrid Search & Re-ranking Pipeline**: Combines semantic search (FAISS) and keyword search (BM25) to retrieve the top 20 candidates, then uses a Cross-Encoder (`BAAI/bge-reranker-base`) to filter for the top 5 most relevant chunks.
  * **Multi-Model Architecture**:
      * **Generator**: Google Gemini 2.5 Flash for fast, context-aware responses.
      * **Judge**: Google Gemini 2.5 Pro acts as an impartial auditor, scoring answers on groundedness, completeness, and citations.
  * **Conflict-Aware Generation**: The system explicitly detects and reports conflicting information found across different documents rather than hallucinating a resolution.
  * **Transparent Provenance**:
      * **Trust Score**: A 1-5 star rating provided by the "Judge" LLM.
      * **Developer Mode**: Inspect raw chunks, source metadata, and data flow.

## 🛠️ Tech Stack

  * **UI Framework**: [Streamlit](https://streamlit.io/)
  * **Orchestration**: LangChain (Community, Google GenAI, HuggingFace integrations)
  * **Vector Store**: FAISS (CPU)
  * **LLMs**: Google Gemini 2.5 Flash & Pro
  * **Embeddings**: `BAAI/bge-base-en-v1.5`
  * **Re-Ranker**: `BAAI/bge-reranker-base`
  * **PDF Processing**: PyMuPDF

## 📋 Prerequisites

  * Python 3.8+
  * A valid **Google Cloud API Key** with access to Gemini models.

## ⚙️ Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/innoscope.git
    cd innoscope
    ```

2.  **Create a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *Dependencies include `streamlit`, `langchain`, `sentence-transformers`, `faiss-cpu`, and `rank_bm25`.*

4.  **Environment Configuration**:
    You can set your API key in a `.env` file (which is git-ignored for security):

    ```bash
    # .env
    GOOGLE_API_KEY="your_api_key_here"
    ```

    *Alternatively, you can enter the key directly in the application sidebar.*

## ▶️ Usage

1.  **Run the application**:

    ```bash
    streamlit run innoscope_app.py
    ```

2.  **Workflow**:

      * **Control Plane (Sidebar)**: Enter your Google API Key if not set in `.env`.
      * **Data Feed**: Upload R\&D Artifacts (PDFs). Click **"Ingest & Index Documents"**.
      * **Query**: Enter a decision query (e.g., "What are the risks of the proposed architecture?").
      * **Review**: Read the generated technical brief, check the "Trust Score," and expand "View Source Evidence" to verify citations.

## 🛡️ Security

  * `.env` files and local PDF data are excluded from version control via `.gitignore` to prevent accidental leakage of secrets or proprietary documents.