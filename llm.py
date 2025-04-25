from langchain_groq import ChatGroq
from vector_database import db
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
model = "Deepseek-r1:1.5b"   #embedding model

query = input("User :")
search_result = db.similarity_search(query)



template = PromptTemplate(
    template="Use the piece of information provided in the given {context} to find the answer to the given {question}. If you don't know, just say you don't know. Don't try to make an answer yourself.",
    input_variables=["context", "question"]
)


chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k":1}),
    chain_type_kwargs = {"prompt":template}
)

result = chain.invoke({"query":query})
print(result["result"])