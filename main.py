from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Literal

from langgraph.graph import StateGraph
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------- TOOL DEFINITIONS ----------

@tool
def get_sts_data(imo: str) -> str:
  """Returns Ship-to-Ship transfer data for the given IMO number."""
  return f"STS data for IMO {imo}: [mock result]"

@tool
def get_gaps_data(imo: str) -> str:
  """Returns AIS gap events for the given IMO number."""
  return f"AIS gaps for IMO {imo}: [mock result]"

# ---------- LLM WITH TOOLS ----------

llm_with_tools = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools([get_sts_data, get_gaps_data])

# ---------- STATE DEFINITION ----------

class GraphState(TypedDict):
  user_input: str
  response: str
  tool_calls: str

# ---------- GRAPH NODES ----------

def input_node(state: GraphState) -> GraphState:
  return {"user_input": state["user_input"]}

def llm_node(state: GraphState) -> GraphState:
  print("Calling LLM with tools...")
  response = llm_with_tools.invoke(state["user_input"])
  return {
    "response": response.content,
    "tool_calls": str(response.tool_calls) if response.tool_calls else "none"
  }

# ---------- BUILD LANGGRAPH ----------

builder = StateGraph(GraphState)
builder.add_node("input", input_node)
builder.add_node("llm", llm_node)
builder.set_entry_point("input")
builder.add_edge("input", "llm")
builder.set_finish_point("llm")
graph = builder.compile()

# ---------- FASTAPI SERVER ----------

app = FastAPI()

class GraphInput(BaseModel):
  user_input: str

@app.post("/run-graph")
async def run_graph(input: GraphInput):
  state = {"user_input": input.user_input}
  result = graph.invoke(state)
  return {
    "llm_output": result["response"],
    "tools_called": result["tool_calls"]
  }
