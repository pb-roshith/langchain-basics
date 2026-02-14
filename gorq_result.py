import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
prompt = ChatPromptTemplate.from_template("Give me a generated name for a {animal}.")
chain = prompt | model | StrOutputParser()

result1 = chain.invoke({"animal": "cat"})
print(f"Result Invoke: {result1}")

inputs = [{"animal":"dog"}, {"animal":"cat"}, {"animal":"dog"}, {"animal":"rat"}]

start = time.time()
result2 = chain.batch(inputs)
end = time.time()

print(f"Results batch: {result2}")
print(f"Time taken: {end - start:.2f} seconds")

for chunk in chain.stream({"animal": "alien"}):
    print(chunk, end="", flush=True)