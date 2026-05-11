import os
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# -----------------------------
# 1. LOAD DOCS (ROBUST)
# -----------------------------
def load_docs():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "ckd_docs.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ CKD docs file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text or not text.strip():
        raise ValueError("❌ CKD docs file is empty")

    return text.strip()


# -----------------------------
# 2. SMART SPLITTER
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)


# -----------------------------
# 3. EMBEDDINGS (OPTIMIZED)
# -----------------------------
def get_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY missing")

    try:
        return GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=api_key
        )
    except Exception as e:
        raise RuntimeError(f"Embedding init failed: {str(e)}")


# -----------------------------
# 4. BUILD VECTORSTORE
# -----------------------------
def build_vectorstore():
    print("🔄 Creating FAISS index...")

    text = load_docs()
    docs = splitter.create_documents([text])

    if len(docs) == 0:
        raise ValueError("❌ No documents created after splitting")

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index")

    print("✅ FAISS index created & saved")

    return vectorstore


# -----------------------------
# 5. LOAD VECTORSTORE (FAST + CACHED)
# -----------------------------
@lru_cache(maxsize=1)
def load_vectorstore():
    embeddings = get_embeddings()

    if os.path.exists("faiss_index"):
        print("📦 Loading FAISS index...")
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    return build_vectorstore()


# -----------------------------
# 6. RETRIEVAL (SMART + SAFE)
# -----------------------------
def retrieve_docs(query: str, k: int = 3):
    if not query or not query.strip():
        return ["⚠️ Empty query received"]

    try:
        vs = load_vectorstore()

        results = vs.similarity_search(query.strip(), k=k)

        if not results:
            return ["⚠️ No relevant context found"]

        # clean output
        return [doc.page_content.strip() for doc in results]

    except Exception as e:
        return [f"❌ Retrieval error: {str(e)}"]