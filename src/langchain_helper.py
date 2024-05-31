import os, pickle, time
from typing import List

import streamlit_helper as sth
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

ASSETS_PATH = os.getcwd() + "/assets"

def process_articles_urls(url_list : List[str], placeholder):
    # load data
    loader = UnstructuredURLLoader(urls=url_list)
    sth.handle_placeholder(placeholder, "Data loading") # UI
    data = loader.load()

    # split data
    text_spliter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ",", ","],
        chunk_size=1000
    )
    sth.handle_placeholder(placeholder, "Text Splitter started")  # UI
    docs = text_spliter.split_documents(data)

    # crate embeddings and save to FAISS index
    embeddings = OpenAIEmbeddings()
    sth.handle_placeholder(placeholder, "Embedding Vector started building")  # UI
    vectorstore_openai = FAISS.from_documents(docs, embedding=embeddings) # local storage vectors
    sth.handle_placeholder(placeholder, "Url processing finished")

    vectorstore_openai.save_local(ASSETS_PATH)


def retrieve_query_from_db(query, urls, placeholder):
    vector_db = FAISS.load_local(ASSETS_PATH, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = OpenAI(temperature=0.7,max_tokens=500)
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever()
    )
    result = chain(
        {"question" : query},
        return_only_outputs=True
    )
    return result