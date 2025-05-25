import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from src.prompt_parser import prompt
load_dotenv()

st.title("⚖️ Legal Assistant Pakistan")
st.image(image="Images/14th-July-Supreme-Court-final-640x360.png")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
def load_vector_store(path,_embedding_model):
    return FAISS.load_local(folder_path=path,embeddings=_embedding_model,allow_dangerous_deserialization=True)
with st.spinner("Loading vector store"):
    path_to_islamic_db = "vector_store/Islamic/"
    path_to_cases_db = "vector_store/Cases/"
    path_to_legal_db = "vector_store/Legal/"

#------------------------------------------Vectors_store------------------------------------------------
    cases_vector_store= load_vector_store(path_to_cases_db,embedding_model)
    legal_vector_store= load_vector_store(path_to_legal_db,embedding_model)
    islamic_vector_store= load_vector_store(path_to_islamic_db,embedding_model)

#-----------------------------------------Retrievers------------------------------------------------------    
cases_retriever = cases_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
legal_retriever = legal_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
islamic_retriever = islamic_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#-------------------------------------CrossEncoder-------------------------------------------------------

hugging_face_reranker = HuggingFaceCrossEncoder(model_name = "cross-encoder/ms-marco-MiniLM-L6-v2")
reranker = CrossEncoderReranker(model=hugging_face_reranker)
pipeline = DocumentCompressorPipeline(transformers=[reranker])

#-----------------------------------------------------------------------------------------------------------
def cleaner(docs):
   return "\n\n".join(doc.page_content for doc in docs)

def main(query,retriever):
    contextual_compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=retriever)

    template = prompt()
    parser = StrOutputParser()
    llm = ChatGroq(model="llama3-70b-8192",max_tokens=512)

    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user","content":query})

    parallel_chain = RunnableParallel({
    "context": contextual_compression_retriever | RunnableLambda(cleaner),
    "question": RunnablePassthrough()})

    final_chain = parallel_chain | template | llm | parser

    with st.spinner("Generating response...."):
        response = final_chain.invoke(query)
        st.chat_message("AI").markdown(response)
        st.session_state.messages.append({"role":"AI","content":response})


st.sidebar.title("Query Selector")
choice = st.sidebar.radio("Choose your Query Type:", ["Legal", "Cases", "Islamic Law"])
query = st.chat_input("Enter your legal query here:")

if query:
    st.sidebar.write(f"You selected: {choice}")
    if choice=="Legal":
        main(query,legal_retriever)
    elif choice=="Cases":
        main(query,cases_retriever)
    elif choice=="Islamic Law":
        main(query,islamic_retriever)
