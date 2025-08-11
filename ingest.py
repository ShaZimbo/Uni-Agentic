""" Process CSV file and build searchable vector database """
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- Load your CSV ---
# Expected columns: 'Question', 'Answer'
df = pd.read_csv('data/faqs.csv')

# Convert each row to a single text block like: "Q: ...\nA: ..."
documents = [
    Document(page_content=f"Q: {row['Question']}\nA: {row['Answer']}")
    for _, row in df.iterrows()
]

# Split into chunks (works even if some answers are long)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# --- Local, free embeddings ---
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

# Build FAISS index (no API calls)
db = FAISS.from_documents(docs, embeddings)

# Save locally for qa_chain.py to load
out_dir = Path('vectorstore')
out_dir.mkdir(parents=True, exist_ok=True)
db.save_local(str(out_dir))
print('✅ Vector store saved to ./vectorstore (index.faiss, index.pkl)')
