from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in .env")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant that can answer questions about the blog post on prompt engineering.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say "I don't know".
Question: {question}
Context: {context}
Answer:"""
)

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)




# llm = HuggingFaceEndpoint(
#     endpoint_url="https://api-inference.huggingface.co/models/bigscience/bloom",
#     huggingfacehub_api_token=hf_token,
#     temperature=0.5
# )

question = "What is prompt Engineering?"

retriever_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n".join([doc.page_content for doc in retriever_docs])

message = prompt.invoke({"question": question, "context": docs_content})
input_text = message.to_string()

answer = llm.invoke(message)
print(answer)
