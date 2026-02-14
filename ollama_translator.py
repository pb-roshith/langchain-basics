from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="gemma3:270m")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Translate the user's text into {style}. Do not explain, just translate."),
    ("human", "{text}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({
    "style": "Gen Z Slang",
    "text": "I am hungry mother."
})

print(response)