from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="deepseek-r1-distill-llama-70b",max_tokens=412)
model = "sentence-transformers/all-MiniLM-L6-v2"
def get_embedding(model):
    return HuggingFaceEmbeddings(model_name=model)
embedding = get_embedding(model)

query = input("User :")
path = "vector_store/faiss_db"
vector_store = FAISS.load_local(path,embedding,allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(type="similarity seacrh", search_kwargs={"k": 3})


template = PromptTemplate(
    template=""" 
        You are a Legal Assistant specialized in Pakistani law. 
        Your task is to provide precise and legally accurate answers based on the provided context. Use the following documents:
        - The Constitution of Pakistan
        - The Pakistan Penal Code
        - The Code of Criminal Procedure
        - Land and Property Laws
        - Criminal Law Amendments

        Instructions:
        - Refer ONLY to the provided context to answer the user's legal question.
        - If the context lacks information to answer, respond with:
        "I don't know based on the given context."
        - Maintain a formal and professional tone at all times.
        - Provide relevant citations from the documents, using brackets like [1], [2], etc.
        
        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """,
    input_variables=["context", "question"]
)


chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs = {"prompt":template})

result = chain.invoke({"query":query})
print(result["result"])


# what is the punishment for the theft