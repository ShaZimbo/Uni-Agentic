
import os
import pandas as pd
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Load FAQs data
current_folder = os.getcwd()
data = pd.read_csv(os.path.join(current_folder, 'University FAQ Agent', 'data', 'faqs.csv'))
questions = data['Question'].tolist()

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key="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"
)

# Create vector store
vector_store = FAISS.from_texts(questions, embeddings)

# Access FAISS index from vector store
faiss_index = vector_store.index  # Assuming `index` is the FAISS index attribute

# Save FAISS index
faiss.write_index(faiss_index, "vectorstore/index.faiss")
