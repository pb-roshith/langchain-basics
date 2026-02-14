import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class CountryData(BaseModel):
    capital: str = Field(description="The capital city")
    currency: str = Field(description="The official currency")
    population: int = Field(description="The approximate population")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

structured_llm = llm.with_structured_output(CountryData)

prompt = ChatPromptTemplate.from_template("Give me facts about {country}.")
chain = prompt | structured_llm

result = chain.invoke({"country": "India"})

print(f"Capital: {result.capital}")
print(f"Population: {result.currency}")
print(f"Population: {result.population}")
print(result)