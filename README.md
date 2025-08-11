### app.py
This Python code creates a simple web application using the Streamlit library, which is designed for building interactive data apps quickly. The script begins by importing Streamlit (aliased as st) and a function called get_answer from a local module named qa_chain.

The app displays a title, "University FAQ AI Assistant," at the top of the page. It then presents a text input box labeled "Ask a question about university services:" where users can type their questions. When a user enters a question, the code calls the get_answer function with the user's input as an argument. The returned answer is then displayed on the page using st.write().

This setup allows users to interactively ask questions about university services and receive AI-generated responses in real time. The code is concise and leverages Streamlit's straightforward API to handle user input and output, making it easy to maintain and extend.

### ingest.py
The code in ingest.py is designed to process a CSV file containing frequently asked questions (FAQs) and their answers, and to build a searchable vector database for use in a question-answering application. It begins by importing necessary libraries, including pandas for data manipulation, Path from pathlib for file path operations, and several components from the LangChain ecosystem for document processing and vector storage.

First, the script loads a CSV file (faqs.csv) that is expected to have columns labeled 'Question' and 'Answer'. Each row from the CSV is converted into a Document object, where the question and answer are combined into a single text block formatted as "Q: ...\nA: ...". This approach ensures that both the question and its answer are treated as a single unit during further processing.

To handle potentially long answers and improve the efficiency of embedding and retrieval, the code uses a RecursiveCharacterTextSplitter to break each document into smaller chunks, with a maximum chunk size of 500 characters and an overlap of 50 characters between chunks. This helps maintain context across chunk boundaries and ensures that the vector representations are manageable in size.

For embedding the text chunks, the script uses the HuggingFaceEmbeddings class with the sentence-transformers/all-MiniLM-L6-v2 model, which is a lightweight and efficient model for generating semantic vector representations of text. These embeddings are then used to build a FAISS (Facebook AI Similarity Search) index, which allows for fast similarity searches without requiring external API calls.

Finally, the vector store is saved locally in a directory called vectorstore, making it available for other parts of the application (such as the question-answering chain) to load and use for retrieving relevant answers based on user queries. The script prints a confirmation message once the vector store is successfully saved. This workflow enables efficient, local, and scalable semantic search over a set of FAQs, forming the backbone of an AI-powered FAQ assistant.

### qa_chain.py
The code in qa_chain.py is responsible for loading a pre-built vector database of university FAQs and providing a function to retrieve the most relevant answer to a user's query using semantic similarity search. It begins by importing the necessary modules: Path from pathlib for file path management, FAISS from langchain_community.vectorstores for efficient vector-based search, and HuggingFaceEmbeddings for generating text embeddings using a HuggingFace transformer model.

The script initializes the same embedding model (sentence-transformers/all-MiniLM-L6-v2) that was used during the ingestion phase, ensuring consistency between how the FAQ data was embedded and how user queries are embedded. It then loads the FAISS vector store from the local vectorstore directory, which contains the indexed FAQ data. The allow_dangerous_deserialization=True flag is required for loading the FAISS index, but it should be used with caution as it can pose security risks if the data is untrusted.

A helper function, _extract_answer, is defined to process the retrieved FAQ text chunks. Since each chunk is formatted as "Q: ...\nA: ...", this function extracts and returns only the answer part, making the output cleaner for the user.

The main function, get_answer, takes a user query and an optional parameter k (defaulting to 2) that specifies how many top results to consider. It performs a similarity search against the vector store using the query. If no relevant results are found, it returns a default message indicating that the answer could not be found. Otherwise, it extracts and returns the answer from the most relevant FAQ chunk. This approach allows the application to efficiently and locally retrieve answers to user questions without relying on external APIs or incurring additional costs.

### Folder Layout
Folder layout:
University FAQ Agent/
├─ app.py
├─ ingest.py
├─ qa_chain.py
├─ requirements.txt
├─ data/
│  └─ faqs.csv
└─ vectorstore/             (created by ingest.py)