# graph_multi.py
from typing import List, Dict
import re

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI

from agents.router_agent import RouterAgent
from agents.asset_agent import AssetSearchAgent
from agents.voyage_agent import VoyageInsightsAgent

# import once at module scope (avoid re-imports inside functions)
from session_store import context_system_prompt, update_ctx_from_tool_calls, SESSION_CTX

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

# ---------- secondary routing safeguard (mixed-intent detector) ----------
_VOYAGE_HINTS = re.compile(
    r"\b(sts|ship[-\s]*to[-\s]*ship|gaps?|positional\s+discrepancy|spoof(?:ing)?|zone|port|arrival|departure)\b",
    re.IGNORECASE,
)
_FILTER_HINTS = re.compile(
    r"\b(flag(?:ged)?|with\s+flag|owner|builder|ship\s*type|shiptype|dwt|tonnage|year|built|name|free[-\s]*text)\b",
    re.IGNORECASE,
)

def _looks_like_mixed_intent(uq: str) -> bool:
  if not uq:
    return False
  s = uq.lower()
  return bool(_VOYAGE_HINTS.search(s) and _FILTER_HINTS.search(s))
# -------------------------------------------------------------------------

def run_turn(session_id: str, user_query: str, history: List):
  # 1) Router decides which specialist or clarify
  ctx_system = context_system_prompt(session_id)
  decision = router.decide(ctx_system.content if ctx_system else None, user_query)

  # fallback: if router says clarify but text mixes voyage + filters, force orchestrate
  if (decision.get("route") == "clarify") and _looks_like_mixed_intent(user_query):
    decision = {"route": "orchestrate", "intent": "unknown", "params": {"slots": decision.get("params", {}).get("slots", {})}}

  # 2) Build a human message for the specialist
  user_msg = HumanMessage(user_query)

  # 3) Route
  if decision.get("route") == "asset_search":
    final = asset_agent.run(ctx_system, user_msg, history)

  elif decision.get("route") == "voyage_insights":
    final = voyage_agent.run(ctx_system, user_msg, history)

  elif decision.get("route") == "orchestrate":
    # Step A: search vessels via asset agent
    stepA = asset_agent.run(ctx_system, user_msg, history)
    msgsA = stepA["messages"]

    # Update session ctx so voyage step sees last_vessels immediately
    update_ctx_from_tool_calls(session_id, msgsA)

    # If nothing matched, return asset messages + a short note
    imos = (SESSION_CTX.get(session_id, {}) or {}).get("last_vessels") or []
    if not imos:
      note = AIMessage(content="No vessels matched those filters, so there are no voyage results to show.")
      return msgsA + [note]

    # Step B: run voyage with refreshed context (same natural user turn)
    ctx2 = context_system_prompt(session_id)
    stepB = voyage_agent.run(ctx2, user_msg, history + msgsA)

    # Combine messages from both steps for this turn
    final = {"messages": msgsA + stepB["messages"]}

  else:
    # clarify path: produce a short AI question and stop
    q = "Which vessel (IMO or MMSI) should I use? You can also tell me a flag/filters to search vessels."
    ai = AIMessage(content=q)
    return [ai]  # just return the clarification

  return final["messages"]
