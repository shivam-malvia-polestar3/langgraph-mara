from typing import List, Dict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from agents.router_agent import RouterAgent
from agents.asset_agent import AssetSearchAgent
from agents.voyage_agent import VoyageInsightsAgent

from session_store import context_system_prompt

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# shared summarizer (per-tool summaries)
summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SUMMARY_SYSTEM_PROMPTS = {
  "vessels": "Summarize vessels: count, notable flags, sample names, and any common shiptypes or owner names.",
  "sts": "You are an STS analyst. Summarize STS events succinctly in 1 line.",
  "a_g": "You are an AIS-gaps analyst. Summarize gaps using HOURS; include count, typical durations, and notable start/end points.",
  "z_p": "You analyze zone/port events. Summarize arrivals/departures: top ports/zones and frequency.",
  "p_d": "You analyze positional discrepancies. Summarize likely spoofing: counts, durations, and notable coordinates.",
}

router = RouterAgent()
asset_agent = AssetSearchAgent()
voyage_agent = VoyageInsightsAgent()

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

def run_turn(session_id: str, user_query: str, history: List):
  # 1) Router decides which specialist or clarify
  ctx_system = context_system_prompt(session_id)
  decision = router.decide(ctx_system.content if ctx_system else None, user_query)

  # 2) Build a human message for the specialist
  user_msg = HumanMessage(user_query)

  # 3) Route
  if decision.get("route") == "asset_search":
    final = asset_agent.run(ctx_system, user_msg, history)
  elif decision.get("route") == "voyage_insights":
    final = voyage_agent.run(ctx_system, user_msg, history)
  else:
    # clarify path: produce a short AI question and stop
    q = "Which vessel (IMO or MMSI) should I use? You can also tell me a flag/filters to search vessels."
    ai = AIMessage(content=q)
    return [ai]  # just return the clarification

  return final["messages"]
