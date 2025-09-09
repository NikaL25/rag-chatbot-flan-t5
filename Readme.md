# 🧠 RAG Chatbot: Prompt Engineering Assistant

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Chroma vector database**, and **Google's Flan-T5 model**. It is designed to answer questions specifically about [Lilian Weng's blog post on prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/).

## ✨ What It Does

- Retrieves relevant context chunks from a preprocessed technical blog post
- Generates answers using a fine-tuned seq2seq language model (`google/flan-t5-base`)
- Answers are generated based on real content, improving accuracy and factuality
- Provides a simple web UI via **Streamlit**

---

## 🧩 Technologies Used

| Component              | Description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| `LangChain`            | Framework for chaining language model logic                               |
| `Chroma`               | Lightweight and local vector database                                     |
| `Flan-T5`              | Pre-trained T5-based model fine-tuned by Google for instruction following |
| `SentenceTransformers` | Used for embedding and vector similarity search                           |
| `Streamlit`            | For building an interactive web interface                                 |
| `Transformers`         | HuggingFace pipelines and models                                          |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/NikaL25/rag-chatbot-flan-t5.git
cd rag-chatbot-flan-t5




2. Set Up Environment

Make sure you have Python 3.9 or higher.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Required Python packages (if requirements.txt is not present):

pip install streamlit langchain transformers sentence-transformers chromadb bs4 python-dotenv


3. Set HuggingFace API Token (if needed)

If you want to use HuggingFace's hosted models instead of local inference, create a .env file:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here


🛠️ Build the Vector Store

Run the data ingestion script to fetch and process the blog post:

python rag_data.py


This will:

Load the blog content using BeautifulSoup

Split it into chunks

Embed it using all-MiniLM-L6-v2

Store it in a local Chroma DB (./chroma_db)

💬 Run the Chatbot App

To start the Streamlit app:
streamlit run main.py


You can now open the web interface at http://localhost:8501 and start asking questions related to prompt engineering.

📦 Example CLI Usage

You can also test the RAG pipeline via the command line:

python query_rag.py


This will:

Query the vector store for similar documents

Use Flan-T5 to generate an answer

Print the final response to the console

🧠 Prompt Template

The system prompt used:

You are a helpful assistant that can answer questions about the blog post on prompt engineering.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say "I don't know".
Question: {question}
Context: {context}
Answer:


📁 Project Structure

├── main.py             # Streamlit app for user interaction
├── query_rag.py        # CLI tool to test the RAG pipeline
├── rag_data.py         # Script to load, split, and embed the source blog post
├── chroma_db/          # Directory where Chroma vector DB is persisted
├── .env                # HuggingFace token (optional)
└── README.md


🔍 Sources

Blog post: Prompt Engineering Guide by Lilian Weng

Model: google/flan-t5-base

Embeddings: sentence-transformers/all-MiniLM-L6-v2

📌 Notes

The context is limited to the blog post used for embedding. The bot cannot answer general questions outside that scope.

If the context is insufficient, the bot will respond with "I don't know".
```
