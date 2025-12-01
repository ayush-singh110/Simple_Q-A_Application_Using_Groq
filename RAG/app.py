import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

##load GROQ API Key
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_messages(
    
    [("system","""Answer the questions based on the provided context only.
    Please provide the most accuract response based on the question
    <context>
    {context}
    </context>"""),
    ("user","Question:{input}")
    ]
)

st.title("RAG Document Q&A")

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader=PyPDFLoader(r"C:\genaipro\qna\Research\Attention.pdf")  ##Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ##Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

    return st.session_state.vectors


user_prompt=st.text_input("Enter your query from the research paper")



if user_prompt:
    vectors=create_vector_embeddings()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({"input":user_prompt})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-------------------------------')
