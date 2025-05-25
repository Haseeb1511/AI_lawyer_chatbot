from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()



def doc_loader(path):
    loader = DirectoryLoader(path=path, glob="*.pdf",loader_cls=PyMuPDFLoader)
    load = loader.load()
    return load

path_to_cases = "pdf_files/cases"
path_to_legal = "pdf_files/Constitution and law"
path_to_islamic = "pdf_files/Islamic law"

cases_document = doc_loader(path_to_cases)
legal_document = doc_loader(path_to_legal)
islamic_law_document = doc_loader(path_to_islamic)

print("len of islamic document is :",len(islamic_law_document))
print("len of cases document is :",len(cases_document))
print("len of legal document is :",len(legal_document))



#-----------------------------------------------Text Splitting-----------------------------------------------------------------------------------

# Step 2: Text splitting
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250
    )
    return text_splitter.split_documents(data)

legal_chunks = text_splitter(legal_document)
islmaic_chunks = text_splitter(islamic_law_document)
cases_chunks = text_splitter(cases_document)

print("Total Chunks legal:",len(legal_chunks))
print("Total Chunks cases:",len(cases_chunks))
print("Total Chunks islamic:",len(islmaic_chunks))

if __name__=="__main__":
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    batch_size = 50

    def create_vector_store(chunks,vector_store_save_path,batch_size):
        vector_store = None
        for i in tqdm(range(0,len(chunks),batch_size)):
            batch = chunks[i:i+batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(documents=batch,embedding=embedding_model)
            else:
                new_store = FAISS.from_documents(documents=batch,embedding=embedding_model)
                vector_store.merge_from(new_store)

        vector_store.save_local(vector_store_save_path)
        return vector_store

    os.makedirs("vector_store/Islamic",exist_ok=True)
    os.makedirs("vector_store/Cases",exist_ok=True)
    os.makedirs("vector_store/Legal",exist_ok=True)

    path_to_islamic_db = "vector_store/Islamic/"
    path_to_cases_db = "vector_store/Cases/"
    path_to_legal_db = "vector_store/Legal/"


    islamic_vector_store = create_vector_store(islmaic_chunks,path_to_islamic_db,batch_size)
    legal_vector_store = create_vector_store(legal_chunks,path_to_legal_db,batch_size)
    cases_vector_store = create_vector_store(cases_chunks,path_to_cases_db,batch_size)

    print("Vector stroe created successfully")