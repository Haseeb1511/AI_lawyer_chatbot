from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv("../.env")  


path = "pdf_files/"
def loader(path):
    loader = DirectoryLoader(path=path, glob="*.pdf",loader_cls=PyPDFLoader)
    load = loader.load()
    return load

loaded_pdf_file = loader(path)
print("Length of pdf: ",len(loaded_pdf_file))

# Step 2: Text splitting
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )
    return text_splitter.split_documents(data)
chunks = text_splitter(data=loaded_pdf_file)
print("Total Chunks:",len(chunks))


#Step 3: Vector
model = "sentence-transformers/all-MiniLM-L6-v2"
def get_embedding(model):
    return HuggingFaceEmbeddings(model_name=model)
embedding = get_embedding(model)

path_to_db = "vector_store/faiss_db"
db = FAISS.from_documents(documents=chunks,embedding=embedding)
db.save_local(path_to_db)

