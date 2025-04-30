import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Legal Assistant Pakistan",layout="wide")
st.title("⚖️ Legal Assistant Pakistan")

st.markdown("""
#### 📄 Legal Documents Included in This Assistant:
Below is the list of legal references used to provide accurate and context-aware responses:
1. **Code of Criminal Procedure (Act V of 1898)**  
   Covers the procedural aspects of criminal law in Pakistan.
2. **The Constitution of the Islamic Republic of Pakistan**  
   The supreme law of Pakistan outlining fundamental rights, governance structure, and legal framework.
3. **Pakistan Penal Code (Act XLV of 1860)**  
   Defines criminal offenses and prescribes punishments applicable throughout Pakistan.
4. **A Guide on Land and Property Rights in Pakistan**  
   Practical guide to understanding land ownership, tenancy, and property laws in Pakistan.
5. **The Pakistan Criminal Law Amendment Act, 1958 (XL of 1958)**  
   Amendments to criminal law aimed at enhancing legal procedures and accountability.
---
🧠 This assistant retrieves answers directly from the above documents using vector-based search and AI-powered reasoning.
""")



if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])
#------------------------------------------------------------------------------------------------------------

llm = ChatGroq(model="deepseek-r1-distill-llama-70b",max_tokens=512)
model = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(model):
    return HuggingFaceEmbeddings(model_name=model)
embedding = get_embedding(model)

path = "vector_store/faiss_db"
vector_store = FAISS.load_local(path,embedding,allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(type="similarity seacrh", search_kwargs={"k": 3})

template = PromptTemplate(
            template="""You are a highly accurate assistant.
                Use ONLY the given context to answer the user's question.
                If the context does not contain the information needed, simply reply:
                "I don't know based on the given context."
                CONTEXT:
                {context}
                QUESTION:
                {question}
                Your Answer:
                Your Answer (with citations like [1], [2]):
                """,
        input_variables=["context", "question"])


query = st.chat_input("Enter your legal query here:")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user","content":query})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs = {"prompt":template})
    with st.spinner("Generating response....."):
        result = chain.invoke({"query":query})
        response = result["result"].replace("<think>","").strip()
        st.chat_message("AI").markdown(response)
        st.session_state.messages.append({"role":"AI","content":response})


