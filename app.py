import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

##Langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT_V2"]="true"

llm=ChatGroq(model="llama-3.1-8b-instant")
## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,temperature):
    llm=ChatGroq(model="llama-3.1-8b-instant",temperature=temperature)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer

st.title("Enhanced Q&A Chatbot")

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)

st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,temperature)
    st.write(response)
else:
    st.write("Please Provide the query")