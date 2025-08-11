from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import tools_condition

from tools_asset_info import get_vessel_data
from tools_voyage import get_ais_gaps, get_sts_data, get_zone_port_events, get_positional_discrepancy

# The full toolset
TOOLS = [get_vessel_data, get_ais_gaps, get_sts_data, get_zone_port_events, get_positional_discrepancy]
TOOL_MAP = {t.name: t for t in TOOLS}

# LLMs
planner = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Per-tool summaries (expanded to include 'vessels')
SUMMARY_SYSTEM_PROMPTS: Dict[str, str] = {
  "vessels": "Summarize vessels: count, notable flags, sample names, and any common shiptypes or owner names.",
  "sts": "You are an STS analyst. Summarize STS events succinctly in 1 line.",
  "a_g": "You are an AIS-gaps analyst. Summarize gaps using HOURS as the unit; include count, typical durations, and notable start/end locations.",
  "z_p": "You analyze zone/port events. Summarize arrivals/departures: top ports/zones and frequency.",
  "p_d": "You analyze positional discrepancies. Summarize likely spoofing: counts, durations, and notable coordinates.",
}

# ====== HARDENED planner policy ======
# Key changes:
# - NEVER call get_vessel_data if an IMO is available in session defaults.
# - If no IMO and no last_vessels are available -> ASK A CLARIFYING QUESTION (do not call any tool).
# - Prefer explicit arguments over guesses; use 'min_duration_hours' when user hints exist.
# - Respect prior window; only change if user overrides.
PLANNER_BASE_SYSTEM = (
  "You are a maritime analytics assistant. Follow these rules strictly:\n"
  "1) Context & defaults:\n"
  "   • Use session defaults (default_imo/default_mmsi/default_time_window) and last_vessels when the user omits them.\n"
  "   • If default_imo exists, DO NOT call get_vessel_data.\n"
  "   • If last_vessels exists and the user asks for STS/GAPS/ZONE/PD 'for those' or similar, run those tools per IMO from last_vessels.\n"
  "2) Clarify instead of assuming:\n"
  "   • If neither default_imo nor last_vessels is available and the user asks for STS/GAPS/ZONE/PD, ASK a short clarifying question like "
  "     'Which vessel (IMO/MMSI)?' Do NOT call get_vessel_data unless the user explicitly asks to search/find vessels.\n"
  "   • Only call get_vessel_data when the user explicitly wants to search (mentions flag/name/owner/builder/shiptype/free-text vessel search).\n"
  "3) Chaining:\n"
  "   • Users may pivot: vessels → STS → GAPS → positional discrepancy → zone/port, etc. Maintain the same IMO(s) and time window unless overridden.\n"
  "   • Respect 'top N' selection if provided; otherwise limit fan-out to reasonable batches (≤ 15 IMOs) when using last_vessels.\n"
  "4) Arguments mapping:\n"
  "   • Always pass timestamp_start/timestamp_end if known; otherwise let tools default.\n"
  "   • Pass MMSI when available.\n"
  "   • If the user says 'duration greater than X hours', pass min_duration_hours=X to the relevant tool.\n"
  "5) Tool selection hygiene:\n"
  "   • If the user asks for 'gaps', only call get_ais_gaps (and only for the intended vessel(s)).\n"
  "   • If the user asks for 'sts', only call get_sts_data (… likewise for z_p and p_d).\n"
  "   • If the user asks to 'search/find vessels', then (and only then) call get_vessel_data with explicit filters.\n"
  "6) Complete the plan first, then reply. Prefer issuing all needed tool calls in a single turn.\n"
  "7) If essential information is still missing after reading the context_system_prompt, respond with a clarifying question instead of calling any tools.\n"
)

def _run_tools_concurrently(tool_calls: List[dict]) -> List[ToolMessage]:
  msgs: List[ToolMessage] = []

  def _invoke_one(call):
    name = call.get("name")
    args = call.get("args") or {}
    call_id = call.get("id")
    tool = TOOL_MAP.get(name)
    if not tool:
      return ToolMessage(content='{"type":"unknown","error":"Unknown tool"}', name=name or "unknown", tool_call_id=call_id)
    try:
      result = tool.invoke(args)
    except Exception as e:
      result = {"type": name, "error": f"{type(e).__name__}: {str(e)}"}
    import json
    return ToolMessage(content=json.dumps(result), name=name, tool_call_id=call_id)

  with ThreadPoolExecutor(max_workers=max(1, len(tool_calls))) as ex:
    futures = [ex.submit(_invoke_one, c) for c in tool_calls]
    for f in as_completed(futures):
      msgs.append(f.result())
  return msgs

def tools_node(state: MessagesState):
  last = state["messages"][-1]
  tcs = getattr(last, "tool_calls", None) or []
  return {"messages": _run_tools_concurrently(tcs)}

def planner_node(state: MessagesState):
  ai = planner.invoke(state["messages"])
  return {"messages": [ai]}


# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("planner", planner_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", tools_condition)
builder.add_edge("tools", "planner")
graph = builder.compile()

def summarize_per_tool(groups: Dict[str, List[dict]]) -> Dict[str, str]:
  from concurrent.futures import ThreadPoolExecutor, as_completed
  out: Dict[str, str] = {}

  def _summ(typ: str, payload: List[dict]):
    sys = SUMMARY_SYSTEM_PROMPTS.get(typ, "Summarize succinctly for a maritime analyst.")
    smsg = SystemMessage(content=sys)
    import json
    hmsg = HumanMessage(content=f"DATASET_TYPE={typ}\nJSON:\n{json.dumps(payload, ensure_ascii=False)}")
    msg = summarizer.invoke([smsg, hmsg])
    return typ, (msg.content if hasattr(msg, "content") else str(msg))

  filtered = {typ: [x for x in payload if "error" not in x] for typ, payload in groups.items()}
  filtered = {typ: v for typ, v in filtered.items() if v}

  if not filtered:
    return out

  with ThreadPoolExecutor(max_workers=max(1, len(filtered))) as ex:
    futures = [ex.submit(_summ, typ, payload) for typ, payload in filtered.items()]
    for f in as_completed(futures):
      typ, text = f.result()
      out[typ] = text
  return out
