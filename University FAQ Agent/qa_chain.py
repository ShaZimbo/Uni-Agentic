from langchain import OpenAI
from langchain.vectorstores.faiss import FAISS

# Load vector store
vector_store = FAISS.load("vectorstore/index.faiss")

def get_answer(question):
    # Generate answer using retrieval-augmented generation (RAG)
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = OpenAI.execute(question, context)
    return answer
