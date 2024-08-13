import streamlit as st
from langchain.document_loaders import NewsURLLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import re
from key import key
def text_cleaner(text):
    clean = re.sub(r'[^a-zA-Z0-9,.""! ]',' ',text)
    return clean
os.environ['OPENAI_API_KEY']=key
st.header('News.AI')
query_label =""
input_label = st.text_input('Enter URL')
docs =""
if input_label:
    url = [f'{input_label}']
    loader = NewsURLLoader(urls=url)
    docs = loader.load()
    st.markdown('### Page Content')
    st.write(docs[0].page_content)

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )

    texts = text_splitter.split_documents(docs)
    llm = OpenAI(temperature=0.6)
    db = FAISS.from_documents(texts,OpenAIEmbeddings())
    chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever = db.as_retriever(search_kwargs={"k":2}),
    )

    st.markdown('### Enter your Queries')
    with st.form(key='query_form'):
        query_label = st.text_input("Enter your query here...")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        if query_label:
            response = chain.run(query_label)
            res = text_cleaner(response)
            st.write(f'Answer: {res}')
        
        

    st.markdown('### Source')
    st.write(input_label)     
         
             







