import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # <-- CHANGED: Local Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="InnoScope POC", layout="wide", page_icon="🔭")

# --- HEADER ---
st.title("🔭 InnoScope")
st.markdown("""
**Retrieval-Augmented Decision Intelligence for Software R&D** *Powered by Google Gemini 2.5 Pro + Local Embeddings*
""")

# --- SIDEBAR: SETUP & INGESTION ---
with st.sidebar:
    st.header("⚙️ Control Plane")

    # 1. API Key Handling
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.text_input("Google API Key", type="password")

    if not api_key:
        st.warning("Please enter a Google API Key to proceed.")
        st.markdown("[Get a Key from Google AI Studio](https://aistudio.google.com/)")
        st.stop()
    else:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    # 2. File Upload
    st.header("📂 Data Feed")
    st.info("Upload Design Docs, ADRs, or Patents (PDF)")
    uploaded_files = st.file_uploader(
        "Upload R&D Artifacts",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button("Ingest & Index Documents")


# --- CORE LOGIC: INGESTION PIPELINE ---
def process_documents(uploaded_files):
    """
    1. Loads PDFs using PyMuPDF.
    2. Splits text into chunks.
    3. Embeds chunks using LOCAL HuggingFace model (No Rate Limits).
    4. Stores vectors in FAISS.
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

    # Step C: Embedding & Indexing
    # USING LOCAL MODEL to avoid Google Rate Limits
    status_text.text("Generating Embeddings (Local CPU - 'BAAI/bge-base-en-v1.5')...")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(splits, embeddings)

    status_text.text("Ready!")
    return vectorstore


# --- SESSION STATE ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if process_button and uploaded_files:
    with st.spinner("Running Ingestion Pipeline (This may take a moment on CPU)..."):
        try:
            st.session_state.vectorstore = process_documents(uploaded_files)
            st.success(f"Indexed {len(uploaded_files)} documents successfully.")
        except Exception as e:
            st.error(f"Error during ingestion: {e}")

# --- CORE LOGIC: RETRIEVAL PIPELINE ---
if st.session_state.vectorstore:
    st.divider()

    # User Input Area
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Enter your decision query:",
                              placeholder="e.g., What are the risks of the proposed architecture?")

    if query:
        # 1. Retriever
        retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        # 2. LLM: Switched to gemini-2.5-flash-lite (Latest)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.2,
            convert_system_message_to_human=True
        )

        # 3. System Prompt
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

        # 4. Run the Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        with st.spinner("Consulting the knowledge base..."):
            response = rag_chain.invoke({"input": query})

        # --- DISPLAY RESULTS ---
        st.markdown("### 📝 Generated Brief")
        st.markdown(response["answer"])

        # Provenance Viewer
        st.markdown("---")
        with st.expander("🔍 View Source Evidence (Provenance)"):
            for i, doc in enumerate(response["context"]):
                st.markdown(
                    f"**Reference {i + 1}:** `{doc.metadata.get('source_file', 'Unknown')}` (Page {doc.metadata.get('page', 0) + 1})")
                st.caption(doc.page_content)
                st.divider()

else:
    st.info("👈 Please upload documents in the sidebar to initialize the engine.")