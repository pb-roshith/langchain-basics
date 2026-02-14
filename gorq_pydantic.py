import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Travel(BaseModel):
    city: str = Field(description="The name of the city the user wants to visit")
    activity_type: str = Field(description="The type of activity: 'food', 'sightseeing', or 'shopping'")
    budget: int = Field(description="The maximum amount of money to spend in USD")

@tool("travel_planner", args_schema=Travel)
def get_travel_recommendation(city: str, activity_type: str, budget:int) -> str:
    """Provides a specific travel suggestion based on city, interest, and budget.""" 

    if budget < 50:
        return f"Economy Option in {city}: Walk around the park and eat street food."
    elif budget < 200:
        return f"Standard Option in {city}: Visit the {activity_type} center and have a nice dinner."
    else:
        return f"Luxury Option in {city}: Private {activity_type} tour with champagne."   
    
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

llm_with_tools = llm.bind_tools([get_travel_recommendation])

query = "I want to do some shopping in Paris. I have about 300 dollars."

response = llm_with_tools.invoke([HumanMessage(content=query)])

if response.tool_calls:
    for tool_call in response.tool_calls:
        args = tool_call["args"]
        print(f"AI extracted: {args}") 

        result = get_travel_recommendation.invoke(args)
        print(f"Recommendation: {result}")