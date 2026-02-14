from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="gemma3:270m")

prompt = ChatPromptTemplate.from_template(
    "you are a food expert. explain {topic} to a kid in one sentence."
)

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({"topic": "gauva"})

print(response)