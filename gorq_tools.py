from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def multiply(a:int, b:int) -> int:
    """Multiplies two integers together."""
    return a * b

@tool
def addition(a:int, b:int) -> int:
    """addition of two numbers."""
    return a + b

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_retries=2,
)

llm_with_tools = llm.bind_tools([multiply, addition])

query = "What is 5 + 98 ?"

response = llm_with_tools.invoke([HumanMessage(content=query)])

if response.tool_calls:
    for tool_call in response.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        
        if name == "multiply":
            result = multiply.invoke(args)
            print(f"Result: {result}")
            
        elif name == "addition":
            result = addition.invoke(args)
            print(f"Result: {result}")

