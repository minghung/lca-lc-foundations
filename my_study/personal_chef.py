# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from pprint import pprint
from tavily import TavilyClient
from typing import Dict, Any
from ipywidgets import FileUpload
from IPython.display import display
import base64


client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """search the web for information"""
    return client.search(query)

web_search.invoke("what dish can I make with chicken, rice, and broccoli?")

agent = create_agent(model="gpt-5-nano", tools=[web_search], checkpointer=InMemorySaver(),
                     system_prompt="You are a personal chef that can help me find recipes online based on ingredients I have.")

uploader = FileUpload(accept=".jpg", multiple=False)
display(uploader)

content_mv = uploader.value[0]["content"]
img_bytes = bytes(content_mv)
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

multipart_message = HumanMessage(content=[
    {"type": "text", "text": "Tell me what food found in this image."},
    {"type": "image", "base64": img_b64, "mime_type": "image/jpeg"}
])

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke({"messages": [multipart_message]}, config=config)
pprint(response["messages"][-1].content)


question = HumanMessage(content="what dish can I make with food found in my fridge?")
response = agent.invoke({"messages": [question]}, config=config)
pprint(response["messages"][-1].content)
