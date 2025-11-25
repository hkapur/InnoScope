import streamlit as st
import os
import tempfile
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="InnoScope POC", layout="wide", page_icon="🔭")

# --- HEADER ---
st.title("🔭 InnoScope")
st.markdown("""
**Retrieval-Augmented Decision Intelligence for Software R&D**
*Engine: Gemini 2.5 Flash | Retrieval: Hybrid (Vector + Keyword) | Eval: LLM-Judge*
""")

# --- SIDEBAR: CONTROL PLANE ---
with st.sidebar:
    st.header("⚙️ Control Plane")

    # 1. API Key Handling
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.text_input("Google API Key", type="password")

    if not api_key:
        st.warning("Please enter a Google API Key to proceed.")
        st.stop()
    else:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    # 2. File Upload
    st.header("📂 Data Feed")
    uploaded_files = st.file_uploader(
        "Upload R&D Artifacts",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button("Ingest & Index Documents")


# --- COMPONENT 1: LLM-AS-A-JUDGE ---
def run_llm_as_a_judge(query, answer, context_text):
    """
    Uses a separate LLM call to grade the RAG output.
    """
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    judge_prompt = f"""
    You are an impartial technical judge. Evaluate the following RAG interaction.

    USER QUERY: {query}
    GENERATED ANSWER: {answer}
    CONTEXT PROVIDED: {context_text}

    Criteria:
    1. Groundedness: Is the answer derived ONLY from the context provided?
    2. Completeness: Did it answer the user's specific question?
    3. Citations: Did the answer include citations (e.g., [Source: X])?

    Output a strictly formatted JSON string with these keys:
    "score": (integer 1-5),
    "reasoning": (short explanation)
    """

    try:
        result = judge_llm.invoke(judge_prompt)
        # Attempt to clean markdown code blocks if Gemini adds them
        clean_content = result.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_content)
    except Exception as e:
        return {"score": 0, "reasoning": f"Judge Error: {str(e)}"}


# --- COMPONENT 2: INGESTION PIPELINE (HYBRID) ---
def process_documents(uploaded_files):
    """
    Loads PDFs, chunks them, and builds TWO retrievers:
    1. FAISS (Vector Semantic Search)
    2. BM25 (Keyword Exact Match)
    Returns an EnsembleRetriever that combines both.
    """
    documents = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step A: Load and Parse
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Parsing: {file.name}...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = file.name

        documents.extend(docs)
        os.remove(tmp_path)
        progress_bar.progress((i + 1) / len(uploaded_files))

    # Step B: Chunking
    status_text.text("Chunking content...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Step C: Vector Indexing (Local BAAI Model)
    status_text.text("Building Vector Index (BAAI/bge-base-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Step D: Keyword Indexing (BM25)
    status_text.text("Building Keyword Index (BM25)...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    # Step E: Ensemble (Hybrid)
    status_text.text("Creating Hybrid Ensemble...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]  # Equal weight to keywords and meaning
    )

    return ensemble_retriever


# --- SESSION STATE ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if process_button and uploaded_files:
    with st.spinner("Running Hybrid Ingestion Pipeline..."):
        try:
            st.session_state.retriever = process_documents(uploaded_files)
            st.success(f"Indexed {len(uploaded_files)} documents using Hybrid Search.")
        except Exception as e:
            st.error(f"Error during ingestion: {e}")

# --- COMPONENT 3: RETRIEVAL & GENERATION ---
if st.session_state.retriever:
    st.divider()

    query = st.text_input("Enter your decision query:",
                          placeholder="e.g., What are the risks of the proposed architecture?")

    if query:
        # 1. LLM Setup
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.2,
            convert_system_message_to_human=True
        )

        # 2. System Prompt
        system_prompt = (
            "You are InnoScope, an R&D research assistant. "
            "Use only the provided context to answer the decision query. "
            "You must cite your sources. Every claim should be followed by [Source: Filename, Page X]. "
            "If the context does not contain the answer, strictly reply: 'Need more information.' "
            "Format the output as a professional technical brief."
            "\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # 3. Execution Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

        with st.spinner("Analyzing documents..."):
            response = rag_chain.invoke({"input": query})

        # --- DISPLAY RESULTS ---
        st.markdown("### 📝 Generated Brief")
        st.markdown(response["answer"])

        # --- JUDGE EVALUATION ---
        with st.spinner("🧑‍⚖️ The Judge is auditing this response..."):
            # Concatenate context for the judge to read
            full_context = "\n".join([doc.page_content for doc in response["context"]])
            grade = run_llm_as_a_judge(query, response["answer"], full_context)

            # Display Score
            score = grade.get("score", 0)
            color = "green" if score >= 4 else "orange" if score == 3 else "red"
            st.markdown(f"**Trust Score:** :{color}[{'★' * score}] ({score}/5)")
            st.caption(f"**Judge's Reasoning:** {grade.get('reasoning', 'No reasoning provided')}")

        # --- PROVENANCE ---
        st.markdown("---")
        with st.expander("🔍 View Source Evidence (Provenance)"):
            for i, doc in enumerate(response["context"]):
                st.markdown(
                    f"**Reference {i + 1}:** `{doc.metadata.get('source_file', 'Unknown')}` (Page {doc.metadata.get('page', 0) + 1})")
                st.caption(doc.page_content)
                st.divider()

else:
    st.info("👈 Please upload documents in the sidebar to initialize the engine.")