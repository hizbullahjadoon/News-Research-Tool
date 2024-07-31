# -*- coding: utf-8 -*-

"""
Created on Tue Jul 16 14:00:37 2024

@author: USER
"""

import streamlit as st
from io import StringIO
#from langchain import OpenAI
from langchain.llms import GooglePalm #, OpenAI
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import TextLoader
#from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
import streamlit as st
from io import StringIO
import pprint
import google.generativeai as palm

google_api_key = "Ypur API Key"
palm.configure(api_key=google_api_key)
llm = GooglePalm(temperature = 0.7, google_api_key=google_api_key)
st.title("News Research Tool")
st.sidebar.title("Files")
query = ""
result = {"query":" ", "result":" "}

with st.sidebar:
    # File uploader widget in the sidebar
    
    
    url = st.text_input("Enter URL: ")
    query = st.text_input("Question: ")
    # Button to process the URLs or text file
    process_url_clicked = st.button("Process URL")
    
    # Process the uploaded file when the button is clicked
    if process_url_clicked:
        with st.spinner("Processing URL"):
            if url is not None:
                # Create UnstructuredURLLoader with the URL
                loader = UnstructuredURLLoader(urls=[url])   
                # Load the content from the URL
                data = loader.load()
                docs = data
                r_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', ' '], chunk_size = 200, chunk_overlap = 20)
                #chunks = r_splitter.split_text(textgot)
                split_docs = r_splitter.split_documents(docs)
                st.write(split_docs)
                
                #embeddings = OpenAIEmbeddings()
                embeddings = HuggingFaceInstructEmbeddings()
                vectorIndex = FAISS.from_documents(split_docs,embeddings)
                embeddings = embeddings.embed_documents([query])
                chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever = vectorIndex.as_retriever())
                result = chain(query)
            
                st.success("URL processed successfully!")
st.subheader("Question: ")
st.text(result['query'])
st.subheader("Answer: ")
st.text(result['result'])



