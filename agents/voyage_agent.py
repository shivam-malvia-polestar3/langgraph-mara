# agents/voyage_agent.py

from typing import List
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
  SystemMessage,
  HumanMessage,
  ToolMessage,
  AIMessage,
  AnyMessage,
)
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START

from tools_voyage import get_ais_gaps, get_sts_data, get_zone_port_events, \
  get_positional_discrepancy, get_ais_positions

VOYAGE_POLICY = (
  "You are the Voyage Insights specialist.\n"
  "Use session defaults (IMO/MMSI/time window, last_vessels) and hints (min_duration_hours, selection_limit).\n"
  "Only call the tools explicitly requested (sts, gaps, positional_discrepancy, zone_port). "
  "If 'last_vessels=<list>' is present in context, iterate that exact list (respect selection_limit) and pass the IMO explicitly.\n"
  "Additionally, whenever you call ANY of those tools, also call get_ais_positions exactly once with the same MMSI and time window. "
  "The AIS positions output is logging-only; never include it in user-facing replies.\n"
  "If you lack IMO and last_vessels, DO NOT call any tools; ask a short clarifying question.\n"
  "Never search vessels here; never mass-fanout without explicit user instruction and a selection limit.\n"
)

VOYAGE_TOOLS = [
  get_ais_gaps,
  get_sts_data,
  get_zone_port_events,
  get_positional_discrepancy,
  get_ais_positions,
]

# caps to avoid accidental blowups; require user to specify 'top N' to exceed small limits
MAX_FANOUT_DEFAULT = 10

def _has_default_imo(ctx_text: str) -> bool:
  return "default_imo=" in (ctx_text or "")

def _last_vessels_count(ctx_text: str) -> int:
  # context_system_prompt may include last_vessels_available=<n> if present
  m = re.search(r"last_vessels_available=(\d+)", ctx_text or "")
  return int(m.group(1)) if m else 0

def _selection_limit(ctx_text: str) -> int:
  m = re.search(r"selection_limit=(\d+)", ctx_text or "")
  return int(m.group(1)) if m else 0

class VoyageInsightsAgent:
  """
  Specialist for Voyage Insights. Plans tool calls with an LLM, executes them,
  and (important) runs AIS positions as a **side-effect** for logging only when
  any voyage tool is invoked — without emitting a ToolMessage for it.
  """

  def __init__(self):
    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(VOYAGE_TOOLS)
    self.graph = self._build()

  # ---------- internal helpers ----------
  def _clean_history_for_llm(self, history: List[AnyMessage], max_keep: int = 6) -> List[AnyMessage]:
    """
    Keep a short tail of history for coherence, but:
      - DROP all ToolMessage,
      - DROP any AIMessage that has tool_calls (otherwise OpenAI complains if paired tool responses are missing).
    """
    cleaned: List[AnyMessage] = []
    for m in reversed(history):
      if isinstance(m, ToolMessage):
        continue
      if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
        continue
      mtype = getattr(m, "type", "")
      if mtype in ("human", "ai", "system"):
        cleaned.append(m)
      if len(cleaned) >= max_keep:
        break
    cleaned.reverse()
    return cleaned

  # ---------- graph nodes ----------
  def _planner_node(self, state: MessagesState):
    # Extract the context prompt text (if any)
    ctx_text = ""
    for m in state["messages"]:
      if getattr(m, "type", "") == "system" and "Conversation context defaults and hints:" in (m.content or ""):
        ctx_text = m.content or ""
        break

    # Guard 1: if we have neither default_imo nor last_vessels -> clarify instead of calling any tool
    if not _has_default_imo(ctx_text) and _last_vessels_count(ctx_text) == 0:
      ai = AIMessage(
          content="Which vessel(s) should I use (IMO/MMSI)? If you want me to search, please provide filters (e.g., flag/name/owner)."
      )
      return {"messages": [ai]}

    # Guard 2: if last_vessels is large and no selection_limit -> ask to narrow down to avoid fanout
    lv = _last_vessels_count(ctx_text)
    sel = _selection_limit(ctx_text)
    if lv > 0 and sel == 0 and lv > MAX_FANOUT_DEFAULT:
      ai = AIMessage(
          content=f"I found a set of {lv} vessels from earlier. Please specify 'top N' (e.g., top 5) or give a narrower filter to proceed."
      )
      return {"messages": [ai]}

    # Otherwise, let the LLM plan tool calls normally
    ai = self.llm.invoke(state["messages"])
    return {"messages": [ai]}

  def _tools_node(self, state: MessagesState):
    """
    Execute exactly the tool calls the LLM requested (returning ToolMessages with
    the correct tool_call_id), and ALSO run get_ais_positions once as a **side-effect**
    when any voyage tool is present — WITHOUT returning a ToolMessage for it.
    This avoids OpenAI's 'tool_call_id not found' error.
    """
    last = state["messages"][-1]
    tcs = list(getattr(last, "tool_calls", None) or [])

    # --- SIDE-EFFECT: auto-run AIS positions (no ToolMessage) -----------------
    voyage_names = {"get_sts_data", "get_ais_gaps", "get_positional_discrepancy", "get_zone_port_events"}
    names = {c.get("name") for c in tcs}
    should_side_effect_positions = any(n in voyage_names for n in names) and ("get_ais_positions" not in names)

    if should_side_effect_positions:
      # Try to reuse arguments from one of the voyage tools
      base_args = {}
      for c in tcs:
        if c.get("name") in voyage_names:
          base_args = c.get("args") or {}
          break
      try:
        # Call the tool directly; DO NOT create a ToolMessage
        from tools_voyage import get_ais_positions  # local import avoids circulars
        _ = get_ais_positions.invoke({
          "imo": base_args.get("imo"),
          "mmsi": base_args.get("mmsi"),
          "timestamp_start": base_args.get("timestamp_start"),
          "timestamp_end": base_args.get("timestamp_end"),
        })
        # its return is {"type":"__log_only",...}; intentionally ignored
      except Exception as e:
        # swallow/log; positions are best-effort
        print(f"[AIS-POSITIONS side-effect] {type(e).__name__}: {e}")

    # --- Execute ONLY the LLM-requested tool calls and return ToolMessages ----
    from concurrent.futures import ThreadPoolExecutor, as_completed
    name_to_tool = {t.name: t for t in VOYAGE_TOOLS}

    def _invoke(c):
      name, args, call_id = c.get("name"), c.get("args") or {}, c.get("id")
      tool = name_to_tool.get(name)
      if not tool:
        return ToolMessage(
            content='{"type":"unknown","error":"Unknown tool"}',
            name=name or "unknown",
            tool_call_id=call_id,
        )
      try:
        result = tool.invoke(args)
      except Exception as e:
        result = {"type": name, "error": f"{type(e).__name__}: {str(e)}"}
      return ToolMessage(content=json.dumps(result), name=name, tool_call_id=call_id)

    msgs: List[ToolMessage] = []
    if tcs:
      with ThreadPoolExecutor(max_workers=max(1, len(tcs))) as ex:
        futures = [ex.submit(_invoke, c) for c in tcs]
        for f in as_completed(futures):
          msgs.append(f.result())

    return {"messages": msgs}

  def _build(self):
    builder = StateGraph(MessagesState)
    builder.add_node("planner", self._planner_node)
    builder.add_node("tools", self._tools_node)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", tools_condition)
    builder.add_edge("tools", "planner")
    return builder.compile()

  # ---------- public ----------
  def run(self, context_system: SystemMessage, user_msg: HumanMessage, history: List[AnyMessage]):
    """
    Execute the voyage specialist using a cleaned history to avoid OpenAI
    'tool_calls must be followed by tool responses' issues and to prevent
    accidental re-interpretation of past tool plans.
    """
    msgs: List[AnyMessage] = [SystemMessage(content=VOYAGE_POLICY)]
    if context_system:
      msgs.append(context_system)

    # Clean the history before sending to LLM
    msgs.extend(self._clean_history_for_llm(history))

    # Append the current user turn
    msgs.append(user_msg)

    return self.graph.invoke({"messages": msgs})
