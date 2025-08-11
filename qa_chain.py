""" Similarity search to get nest answer """

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use the same local embedding model as in ingest.py
_embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load the FAISS store created by ingest.py
_vector_path = Path('vectorstore')
_vector_store = FAISS.load_local(
    str(_vector_path),
    _embeddings,
    allow_dangerous_deserialization=True
)


def _extract_answer(text: str) -> str:
    """Given a chunk like 'Q: ...\nA: ...',
    return just the answer part if present."""
    if '\nA:' in text:
        return text.split('\nA:', 1)[1].strip()
    if 'A:' in text:
        return text.split('A:', 1)[1].strip()
    return text.strip()


def get_answer(query: str, k: int = 2) -> str:
    """Retrieve the most relevant chunk(s) and return the best-matching answer.
    This version does NOT call any paid API. It uses similarity search only.
    """
    results = _vector_store.similarity_search(query, k=k)
    if not results:
        return "Sorry, I couldn't find that in the FAQs."
    top = results[0]
    return _extract_answer(top.page_content)
