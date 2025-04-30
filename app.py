import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from vector_database import db
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Legal Assistant Pakistan",layout="wide")
st.title("Legal Assistant Pakistan")

