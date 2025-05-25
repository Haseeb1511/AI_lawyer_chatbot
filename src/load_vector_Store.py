from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

path_to_islamic_db = "vector_store/Islamic/"
path_to_cases_db = "vector_store/Cases/"
path_to_legal_db = "vector_store/Legal/"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_vector_store(path,embedding_model):
    return FAISS.load_local(folder_path=path,embeddings=embedding_model,allow_dangerous_deserialization=True)

cases_vector_store= load_vector_store(path_to_cases_db,embedding_model)
legal_vector_store= load_vector_store(path_to_legal_db,embedding_model)
islamic_vector_store= load_vector_store(path_to_islamic_db,embedding_model)


