from typing import Optional, Dict, Any
import json, re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- Intents & slot rules ---
# We keep it simple: the router decides domain + sub-intent and checks for required slots.
# Required slots for voyage tools: at least one of {IMO, MMSI, last_vessels} must be available.
# If missing -> route=clarify with a targeted question (never backfill via vessel search).
ROUTER_POLICY = (
  "You are a router. Your job is to route each user turn to the correct specialist or ask a clarification.\n"
  "Specialists:\n"
  "- asset_search: when the user explicitly wants VESSEL DETAILS/PARTICULARS or to SEARCH/FILTER vessels "
  "(by IMO/MMSI/name/flag/owner/builder/shiptype/dwt/year ranges, or free-text). Examples: "
  "'get vessel details', 'find vessels with flag Panama', 'list tankers built after 2015'.\n"
  "- voyage_insights: when the user asks for STS, AIS gaps, positional discrepancy (spoofing), or zone/port events "
  "for known vessel(s). Examples: 'give me gaps for IMO 9169055', 'STS for those 5 vessels'.\n"
  "- clarify: when essential info is missing for voyage tools (e.g., no vessel identifiers or prior vessel set).\n\n"
  "Hard rules:\n"
  "1) NEVER use asset_search to backfill missing IMO for a voyage request. If the user says 'give me gaps' "
  "   without a specific vessel and there is no prior vessel set, choose clarify.\n"
  "2) 'details'/'particulars'/'vessel info' => asset_search, even if an IMO is present.\n"
  "3) 'search/find/list/filter vessels' => asset_search.\n"
  "4) Voyage terms ('gaps', 'sts', 'spoofing', 'positional discrepancy', 'zone', 'port', 'arrival', 'departure') "
  "   => voyage_insights IF session has default_imo or last_vessels; otherwise clarify.\n"
  "5) Output strict JSON with keys: route ('asset_search'|'voyage_insights'|'clarify'), intent, params, "
  "   and optional clarify_message.\n"
  "Example: {\"route\":\"voyage_insights\",\"intent\":\"gaps\",\"params\":{\"min_duration_hours\":12}}\n"
)

# Regex helpers
_DETAILS_PAT = re.compile(
    r"(vessel\s+(details?|info|information|particulars?)|particulars?|dimensions?|flag|ownership|owner|"
    r"classification|builder|year\s+of\s+build|ship\s*type|mmsi|name)\b", re.IGNORECASE)

_VOYAGE_HINTS = re.compile(
    r"\b(sts|ship[-\s]*to[-\s]*ship|gaps?|positional\s+discrepancy|spoof(?:ing)?|zone|port|arrival|departure)\b",
    re.IGNORECASE)

_SEARCH_HINTS = re.compile(
    r"\b(search|find\s+vessels?|list\s+vessels?|filter|flag|owner|builder|shiptype|free[-\s]*text)\b",
    re.IGNORECASE)

_IMO_PAT = re.compile(r'\b(?:imo\s*)?([0-9]{7})\b', re.IGNORECASE)
_MMSI_PAT = re.compile(r'\b(?:mmsi\s*)?([0-9]{9})\b', re.IGNORECASE)

def _has_context(context_prompt: Optional[str]) -> bool:
  if not context_prompt:
    return False
  return ("default_imo=" in context_prompt) or ("last_vessels_available=" in context_prompt)

def _extract_slots(user_query: str) -> Dict[str, Any]:
  uq = user_query.lower()
  slots: Dict[str, Any] = {}
  m_imo = _IMO_PAT.search(uq)
  if m_imo:
    slots["imo"] = m_imo.group(1)
  m_mmsi = _MMSI_PAT.search(uq)
  if m_mmsi:
    slots["mmsi"] = m_mmsi.group(1)
  # duration > N hours (hint)
  dur = re.search(r'(?:duration\s*(?:>|>=|greater\s+than)\s*)(\d+)\s*hour', uq, re.IGNORECASE)
  if dur:
    slots["min_duration_hours"] = int(dur.group(1))
  return slots

class RouterAgent:
  def __init__(self):
    self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

  def decide(self, context_prompt: Optional[str], user_query: str) -> Dict[str, Any]:
    uq = user_query.lower()
    slots = _extract_slots(user_query)

    # 1) Explicit vessel details / particulars / attributes -> asset_search
    if _DETAILS_PAT.search(uq):
      return {"route": "asset_search", "intent": "vessel_details", "params": {"slots": slots}}

    # 2) Voyage intents?
    if _VOYAGE_HINTS.search(uq):
      if slots.get("imo") or slots.get("mmsi") or _has_context(context_prompt):
        # route to voyage, pass known hints
        return {"route": "voyage_insights", "intent": "voyage", "params": {"slots": slots}}
      # missing vessel id(s) and no context -> clarify
      return {
        "route": "clarify",
        "intent": "clarify",
        "params": {},
        "clarify_message": "Which vessel should I use (IMO or MMSI)? If you want to search first, tell me the filters (e.g., flag/name/owner)."
      }

    # 3) Generic search phrasing -> asset_search
    if _SEARCH_HINTS.search(uq):
      return {"route": "asset_search", "intent": "vessel_search", "params": {"slots": slots}}

    # 4) If the user mentions a 7-digit IMO and says 'get vessel details' implicitly
    if slots.get("imo"):
      return {"route": "asset_search", "intent": "vessel_details", "params": {"slots": slots}}

    # 5) Model-based fallback bounded by policy
    msgs = [SystemMessage(content=ROUTER_POLICY)]
    if context_prompt:
      msgs.append(SystemMessage(content=context_prompt))
    msgs.append(HumanMessage(content=user_query))
    ai = self.model.invoke(msgs)
    try:
      data = json.loads(ai.content.strip())
      if isinstance(data, dict) and data.get("route") in {"asset_search","voyage_insights","clarify"}:
        return data
    except Exception:
      pass

    # 6) Final fallback: clarify (never mass fan-out)
    return {
      "route": "clarify",
      "intent": "clarify",
      "params": {},
      "clarify_message": "Do you want vessel details or voyage analytics (STS/Gaps/PD/Zone)? If voyage, which vessel(s)?"
    }
