import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

@st.cache_resource
def init_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma(
        collection_name="prompt_engineering",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
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

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that can answer questions about the blog post on prompt engineering.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say "I don't know".
Question: {question}
Context: {context}
Answer:"""
    )   

    return vector_store, llm, prompt

vector_store, llm, prompt = init_resources()

st.title("RAG on flan-t5-base + Chroma + LangChain")
user_question = st.text_input("Ask a question about prompt engineering")

if user_question:
    with st.spinner("Looking for an answer..."):
        docs = vector_store.similarity_search(user_question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        if len(context.strip()) < 50:
            answer = "I don't know"
        else:
            message = prompt.invoke({"question": user_question, "context": context})
            answer = llm.invoke(message)

            if answer.strip().lower() in ["positive", "negative", ""]:
                answer = "I don't know"

    st.subheader("Answer:")
    st.write(answer)

