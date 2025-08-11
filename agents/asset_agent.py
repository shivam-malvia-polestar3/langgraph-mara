from typing import List
import re
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, AnyMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START

from tools_asset_info import get_vessel_data

ASSET_SYSTEM = (
  "You are the Asset Search specialist.\n"
  "Call ONLY get_vessel_data to fetch vessel details from the Asset Information Service (OpenSearch).\n"
  "If the user mentions a 7-digit IMO anywhere, call get_vessel_data with that exact IMO.\n"
  "Otherwise, only search with explicit filters (name/flag/owner/builder/shiptype/dwt/year/free_text).\n"
  "NEVER perform broad/wildcard searches without explicit filters. NEVER call voyage tools here.\n"
)

_IMO_PAT = re.compile(r'\b(?:imo\s*)?([0-9]{7})\b', re.IGNORECASE)


class AssetSearchAgent:
  def __init__(self):
    # Bind only the asset tool here (no voyage tools)
    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([get_vessel_data])
    self.graph = self._build()

  # ---------- internal helpers ----------
  def _clean_history_for_llm(self, history: List[AnyMessage], max_keep: int = 6) -> List[AnyMessage]:
    """
    Keep a short tail of history for coherence, but:
      - DROP all ToolMessage,
      - DROP any AIMessage that has tool_calls (otherwise OpenAI complains if paired tool responses are missing).
    """
    cleaned: List[AnyMessage] = []
    # iterate from the end; collect up to max_keep eligible messages
    for m in reversed(history):
      mtype = getattr(m, "type", "")
      if isinstance(m, ToolMessage):
        continue
      if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
        continue
      # keep only human/ai without tool_calls
      if mtype in ("human", "ai", "system"):
        cleaned.append(m)
      if len(cleaned) >= max_keep:
        break
    cleaned.reverse()
    return cleaned

  # ---------- graph nodes ----------
  def _planner_node(self, state: MessagesState):
    """
    Planner behavior:
    - If the latest user turn contains a 7-digit IMO, synthesize a proper AIMessage
      that directly calls get_vessel_data(imo=...).
    - Otherwise, let the LLM plan a get_vessel_data call using provided filters.
    """
    last_user = None
    for m in reversed(state["messages"]):
      if getattr(m, "type", "") == "human":
        last_user = m.content
        break

    if last_user:
      m = _IMO_PAT.search(last_user or "")
      if m:
        imo = m.group(1)
        # Proper AIMessage with tool_calls so LangGraph routes to the tools node
        ai = AIMessage(
            content="",
            tool_calls=[{
              "id": "auto_get_vessel_data_by_imo",
              "name": "get_vessel_data",
              "args": {"imo": imo}
            }],
        )
        return {"messages": [ai]}

    # No explicit IMO – let the model call get_vessel_data with filters from text
    ai = self.llm.invoke(state["messages"])
    return {"messages": [ai]}

  def _tools_node(self, state: MessagesState):
    """
    Execute tool calls (only get_vessel_data) and return ToolMessages.
    """
    last = state["messages"][-1]
    tcs = getattr(last, "tool_calls", None) or []
    msgs: List[ToolMessage] = []

    for c in tcs:
      name, args, call_id = c.get("name"), c.get("args") or {}, c.get("id")
      try:
        result = get_vessel_data.invoke(args)
      except Exception as e:
        result = {"type": "vessels", "error": f"{type(e).__name__}: {str(e)}"}
      msgs.append(ToolMessage(content=json.dumps(result), name=name, tool_call_id=call_id))

    return {"messages": msgs}

  def _build(self):
    """
    Build the small graph: START -> planner -> (tools?) -> END
    """
    builder = StateGraph(MessagesState)
    builder.add_node("planner", self._planner_node)
    builder.add_node("tools", self._tools_node)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", tools_condition)
    return builder.compile()

  # ---------- public ----------
  def run(self, context_system: SystemMessage, user_msg: HumanMessage, history: List[AnyMessage]):
    """
    Execute the asset specialist with a cleaned, tool-free tail of history
    to avoid OpenAI 'tool_calls must be followed by tool responses' errors.
    """
    msgs: List[AnyMessage] = [SystemMessage(content=ASSET_SYSTEM)]
    if context_system:
      msgs.append(context_system)

    # Clean the history before sending to LLM
    msgs.extend(self._clean_history_for_llm(history))

    # Append the current user turn
    msgs.append(user_msg)

    return self.graph.invoke({"messages": msgs})
