import streamlit as st
import langchain_helper as lch

st.title("News Research Tool")

st.sidebar.title("News Article URLs")

main_placeholder = st.empty()

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

if st.sidebar.button("Process Urls"):
    lch.process_articles_urls(urls, main_placeholder)

query = st.text_area("Question: ")
if st.button("Search"):
    response = lch.retrieve_query_from_db(query, urls, main_placeholder)
    st.write(response["answer"])
    st.write(response['sources'])