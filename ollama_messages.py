from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

llm = ChatOllama(model="gemma3:270m")

messages = [
    SystemMessage(content="You are a sentiment analyzer. Output ONLY: 'Positive', 'Negative', or 'Neutral'. Do not say anything else."),

    HumanMessage(content="you are a bad guy."),
    AIMessage(content="Negative"),

    HumanMessage(content="thanks for your help."),
    AIMessage(content="Positive"),

    HumanMessage(content="he beat his son.")
]

response = llm.invoke(messages)

print(response.content)