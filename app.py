import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from llm import llm
from vector_database import db
from dotenv import load_dotenv
load_dotenv()

st.title("RAG Base AI Lawyer")

upload_file = st.file_uploader("Click to upload file:",
                               type="pdf",
                               accept_multiple_files=True)
query = st.text_area("Enter text here:")
button = st.button("Ask AI")

if button:
    response="this is ai resopnes"
    if query:
        st.chat_message("AI").markdown(response)
        search_result = db.similarity_search(query)

        template = PromptTemplate(
        template="Use the piece of information provided in the given {context} to find the answer to the given {question}. If you don't know, just say you don't know. Don't try to make an answer yourself.",
        input_variables=["context", "question"])

        chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=db.as_retriever(search_kwargs={"k":1}),
                    chain_type_kwargs = {"prompt":template})
                                                    

        result = chain.invoke({"query":query})
        st.write(result["result"])
    else:
        st.warning("Please enter text and upload PDF file first!!")