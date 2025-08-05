# Create directory structure and basic starter files for the AI FAQ assistant project

from pathlib import Path
import pandas as pd

# Base directory
base_dir = Path("/mnt/data/university-faq-agent")

# Create directories
dirs = [
    base_dir,
    base_dir / "data",
    base_dir / "vectorstore"
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

# Define file contents
files = {
    "app.py": """
import streamlit as st
from qa_chain import get_response

st.set_page_config(page_title="University FAQ Assistant")
st.title("🎓 University FAQ Assistant")

st.markdown("Ask me anything about university services, IT support, or academic processes!")

query = st.text_input("Your question:")

if query:
    response = get_response(query)
    st.write("**Answer:**", response)
""",
    "qa_chain.py": """
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_response(query):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(query)
    return result
""",
    "ingest.py": """
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the CSV
df = pd.read_csv("data/faqs_extended.csv")

# Convert to LangChain Document format
documents = [Document(page_content=f"Q: {row['Question']}\\nA: {row['Answer']}") for _, row in df.iterrows()]

# Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Generate embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)

# Save vector DB locally
db.save_local("vectorstore")
print("✅ Vector store saved.")
""",
    ".env": "OPENAI_API_KEY=your_openai_api_key_here\n",
    "requirements.txt": """
openai
langchain
streamlit
python-dotenv
faiss-cpu
pandas
"""
}

# Write files
for filename, content in files.items():
    file_path = base_dir / filename
    file_path.write_text(content.strip(), encoding="utf-8")

# Copy the extended CSV to the new data folder
extended_csv_path = base_dir / "data" / "faqs_extended.csv"
sample_data = [
    {"Question": "What is the university email address?", "Answer": "contact@university.edu"},
    {"Question": "How do I reset my password?", "Answer": "Visit the IT helpdesk portal."}
]
combined_faq_df = pd.DataFrame(sample_data)
combined_faq_df.to_csv(extended_csv_path, index=False)

base_dir.as_posix()
