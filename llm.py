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

chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs = {"prompt":template})

result = chain.invoke({"query":query})
print(result["result"])


# what is the punishment for the theft