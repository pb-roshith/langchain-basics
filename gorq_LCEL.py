from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.9,
    max_tokens=50,
    max_retries=2,
    model_kwargs={
        "top_p":0.8,
        "frequency_penalty": 0.6
    }
)

prompt1 = ChatPromptTemplate.from_template("Explain {topic} in short.")
chain1 = prompt1 | model | StrOutputParser()

prompt2 = ChatPromptTemplate.from_template("translate the text to german : {text}")
chain2 = prompt2 | model | StrOutputParser()

final_chain = {"text": chain1} | chain2

result = final_chain.invoke({"topic": "java"})

print(result)