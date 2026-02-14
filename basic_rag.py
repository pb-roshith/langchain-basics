import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

persist_dir = "./my_chroma_db"

if os.path.exists(persist_dir):
    print("loading existing database")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding)
else:
    print("creating database")
    data = """
        ChromaDB is an open-source vector database. 
        It allows for easy persistence of embeddings.
        The project 'Titan' is being built using Chroma and Llama 3.
        Titan's primary goal is secure, local-first data retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    chunks = splitter.split_text(data)

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        persist_directory=persist_dir 
    )

retriever = vector_store.as_retriever(search_kwargs={"k":2})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context:
{context}

Question: {input}
""")

rag_chain = (
    {
        "context": itemgetter("input") | retriever | format_docs,
        "input": itemgetter("input")
    }
    | prompt
    | llm
    | StrOutputParser()
)

query = "which project is built using Chroma and Llama 3?"
result = rag_chain.invoke({"input":query})

print(result)