from typing import List, Optional, Dict, Any
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

# ============ Flag extraction (robust) ============
_FLAG_BEFORE_FLAGGED = re.compile(r'\b([A-Za-z][A-Za-z\s&\.\'-]{2,40})\s*(?:-\s*)?flagged\b', re.IGNORECASE)
_FLAG_AFTER_FLAGGED  = re.compile(r'\bflagged\s+([A-Za-z][A-Za-z\s&\.\'-]{2,40})\b', re.IGNORECASE)
_FLAG_EQ              = re.compile(r'\bflag(?:\s*(?:is|=|:)?\s*|[\s-]+)([A-Za-z][A-Za-z\s&\.\'-]{2,40})\b', re.IGNORECASE)
_WITH_FLAG            = re.compile(r'\bwith\s+flag\s+([A-Za-z][A-Za-z\s&\.\'-]{2,40})\b', re.IGNORECASE)
_FLAG_OF              = re.compile(r'\bflag\s+of\s+([A-Za-z][A-Za-z\s&\.\'-]{2,40})\b', re.IGNORECASE)

_NEEDS_PREV = {"islands","man","kong","kingdom","states","guinea","emirates","kitts","nevis",
               "tobago","leone","ivoire","zealand","lanka","verde","principe","guadeloupe",
               "guernsey","jersey"}
_DEMONYM = {
  "panamanian": "Panama",
  "bahamian": "Bahamas",
  "liberian": "Liberia",
  "marshallese": "Marshall Islands",
  "hong kong": "Hong Kong",
  "british": "United Kingdom",
  "u.k.": "United Kingdom", "uk": "United Kingdom",
  "u.s.": "United States", "us": "United States", "american": "United States",
  "emirati": "United Arab Emirates", "emirates": "United Arab Emirates",
}
_FILLER = {"all","the","any","every","of","for","under","with","registered","registry",
           "vessel","vessels","ship","ships","tanker","tankers","bulk","carrier","carriers",
           "flag","flagged"}

def _normalize_flag_phrase(raw: str) -> Optional[str]:
  if not raw: return None
  tokens = [t for t in re.findall(r"[A-Za-z']+", raw) if t.lower() not in _FILLER]
  if not tokens: return None
  low = " ".join(tokens).lower().strip()
  if low in _DEMONYM: return _DEMONYM[low]
  if len(tokens) >= 2 and tokens[-1].lower() in _NEEDS_PREV:
    cand = f"{tokens[-2]} {tokens[-1]}"
  else:
    cand = tokens[-1]
  low_cand = cand.lower()
  if low_cand in _DEMONYM: return _DEMONYM[low_cand]
  return " ".join(w[:1].upper() + w[1:] for w in cand.split())

def _extract_flag(text: str) -> Optional[str]:
  if not text: return None
  for pat in (_FLAG_BEFORE_FLAGGED, _FLAG_AFTER_FLAGGED, _FLAG_EQ, _WITH_FLAG, _FLAG_OF):
    m = pat.search(text)
    if m: return _normalize_flag_phrase(m.group(1))
  m = re.search(r"\b([A-Za-z][A-Za-z\s&\.\'-]{2,40})-\s*flagged\b", text, re.IGNORECASE)
  return _normalize_flag_phrase(m.group(1)) if m else None

# ============ Owner / Operator / Manager ============
_OWNER_PATS = [
  re.compile(r'\bowner\s+(?:is|=|:)?\s*["“]?([A-Za-z0-9&\.\'\-\s]{2,60})["”]?', re.IGNORECASE),
  re.compile(r'\bowned\s+by\s+["“]?([A-Za-z0-9&\.\'\-\s]{2,60})["”]?', re.IGNORECASE),
  re.compile(r'\b(operator|manager|managed\s+by|operated\s+by)\s+["“]?([A-Za-z0-9&\.\'\-\s]{2,60})["”]?', re.IGNORECASE),
  re.compile(r'\bowner[:=]\s*([A-Za-z0-9&\.\'\-\s]{2,60})', re.IGNORECASE),
]

def _extract_owner(text: str) -> Optional[str]:
  for p in _OWNER_PATS:
    m = p.search(text)
    if m:
      # handle (operator|manager, name)
      name = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
      return name.strip(' "—-')
  return None

# ============ Builder / Yard ============
_BUILDER_PATS = [
  re.compile(r'\b(builder|yard)\s*(?:is|=|:)?\s*["“]?([A-Za-z0-9&\.\'\-\s]{2,60})["”]?', re.IGNORECASE),
  re.compile(r'\bbuilt\s+by\s+["“]?([A-Za-z0-9&\.\'\-\s]{2,60})["”]?', re.IGNORECASE),
]

def _extract_builder(text: str) -> Optional[str]:
  for p in _BUILDER_PATS:
    m = p.search(text)
    if m:
      name = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
      return name.strip(' "—-')
  return None

# ============ Shiptype ============
# Catch explicit "shiptype=..." or natural mentions like "tanker(s)", "bulk carrier(s)", etc.
_SHIPTYPE_EXPL = re.compile(r'\bship\s*type\s*(?:is|=|:)?\s*([A-Za-z0-9/\-\s]{3,50})', re.IGNORECASE)
# common families
_SHIPTYPE_HINTS = {
  "tanker": "Tanker",
  "product tanker": "Product Tanker",
  "crude tanker": "Crude Oil Tanker",
  "lng": "LNG Tanker",
  "lpg": "LPG Tanker",
  "bulk": "Bulk Carrier",
  "bulker": "Bulk Carrier",
  "container": "Container Ship",
  "ro-ro": "Ro-Ro",
  "roro": "Ro-Ro",
  "general cargo": "General Cargo",
  "offshore": "Offshore",
}

def _extract_shiptype(text: str) -> Optional[str]:
  m = _SHIPTYPE_EXPL.search(text)
  if m:
    return m.group(1).strip(' "—-')
  low = text.lower()
  for hint, norm in _SHIPTYPE_HINTS.items():
    if re.search(rf'\b{re.escape(hint)}(s)?\b', low):
      return norm
  return None

# ============ DWT & Year ranges ============
_NUM = re.compile(r'(\d{1,3}(?:[_,]\d{3})*|\d+(?:\.\d+)?)\s*([km])?', re.IGNORECASE)

def _num_to_int(s: str, suf: Optional[str]) -> int:
  s_clean = s.replace(",", "").replace("_", "")
  val = float(s_clean)
  if suf and suf.lower() == "k": val *= 1_000
  if suf and suf.lower() == "m": val *= 1_000_000
  return int(val)

# DWT: >, >=, <, <=, between
_DWT_BETWEEN = re.compile(r'\b(dwt|deadweight|tonnage)\s*(?:between|from)\s*' + _NUM.pattern + r'\s*(?:to|and|-)\s*' + _NUM.pattern, re.IGNORECASE)
_DWT_MIN = re.compile(r'\b(dwt|deadweight|tonnage)\s*(?:>=|>|at\s*least|more\s+than|over|greater\s+(?:than|then))\s*' + _NUM.pattern, re.IGNORECASE)
_DWT_MAX = re.compile(r'\b(dwt|deadweight|tonnage)\s*(?:<=|<|at\s*most|no\s+more\s+than|under|less\s+than)\s*' + _NUM.pattern, re.IGNORECASE)

# Year of build: after/before/between
_YEAR_BETWEEN = re.compile(r'\b(built|year\s*of\s*build|yob)\s*(?:between|from)\s*(\d{4})\s*(?:to|and|-)\s*(\d{4})\b', re.IGNORECASE)
_YEAR_MIN = re.compile(r'\b(built|year\s*of\s*build|yob)\s*(?:>=|>|after|since|from)\s*(\d{4})\b', re.IGNORECASE)
_YEAR_MAX = re.compile(r'\b(built|year\s*of\s*build|yob)\s*(?:<=|<|before|until|upto|up\s*to)\s*(\d{4})\b', re.IGNORECASE)

def _extract_dwt(text: str) -> Dict[str, int]:
  out: Dict[str, int] = {}
  m = _DWT_BETWEEN.search(text)
  if m:
    lo, lo_suf, hi, hi_suf = m.group(2), m.group(3), m.group(4), m.group(5)
    out["dwt_min"] = _num_to_int(lo, lo_suf); out["dwt_max"] = _num_to_int(hi, hi_suf); return out
  m = _DWT_MIN.search(text)
  if m:
    lo, lo_suf = m.group(2), m.group(3)
    out["dwt_min"] = _num_to_int(lo, lo_suf); return out
  m = _DWT_MAX.search(text)
  if m:
    hi, hi_suf = m.group(2), m.group(3)
    out["dwt_max"] = _num_to_int(hi, hi_suf); return out
  return out

def _extract_year(text: str) -> Dict[str, int]:
  out: Dict[str, int] = {}
  m = _YEAR_BETWEEN.search(text)
  if m:
    out["year_min"] = int(m.group(2)); out["year_max"] = int(m.group(3)); return out
  m = _YEAR_MIN.search(text)
  if m:
    out["year_min"] = int(m.group(2)); return out
  m = _YEAR_MAX.search(text)
  if m:
    out["year_max"] = int(m.group(2)); return out
  return out

# ============ Name & free-text ============
_NAME_PAT = re.compile(r'\bname\s*(?:is|=|:)?\s*["“]?([A-Za-z0-9\s\-\.\'/_]{2,80})["”]?', re.IGNORECASE)
_QUOTED_FREE = re.compile(r'["“]([^"”]{2,120})["”]')

def _extract_name(text: str) -> Optional[str]:
  m = _NAME_PAT.search(text)
  if m: return m.group(1).strip(' "—-')
  return None

def _extract_free_text(text: str) -> Optional[str]:
  # only use if nothing else matched; take the first quoted phrase as a hint
  m = _QUOTED_FREE.search(text)
  if m: return m.group(1).strip()
  return None

def _extract_asset_filters(text: str) -> Dict[str, Any]:
  """Return a dict with any of: name, flag, owner, builder, shiptype, dwt_min, dwt_max, year_min, year_max, free_text."""
  filters: Dict[str, Any] = {}
  if not text: return filters

  flag = _extract_flag(text)
  if flag: filters["flag"] = flag

  owner = _extract_owner(text)
  if owner: filters["owner"] = owner

  builder = _extract_builder(text)
  if builder: filters["builder"] = builder

  shiptype = _extract_shiptype(text)
  if shiptype: filters["shiptype"] = shiptype

  dwt = _extract_dwt(text); filters.update(dwt)
  yr = _extract_year(text); filters.update(yr)

  name = _extract_name(text)
  if name: filters["name"] = name

  # optional free_text only if no primary filters
  if not filters:
    ft = _extract_free_text(text)
    if ft: filters["free_text"] = ft

  return filters

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
    - If latest user turn contains a 7-digit IMO, synthesize a direct call get_vessel_data(imo=...).
    - Else, deterministically parse filters (flag/owner/builder/shiptype/dwt/year/name/free_text).
    - If any filters are found, call get_vessel_data with those args (no LLM guessing).
    - Else, let the LLM plan a get_vessel_data call.
    """
    last_user = None
    for m in reversed(state["messages"]):
      if getattr(m, "type", "") == "human":
        last_user = m.content or ""
        break

    # 1) direct IMO – always wins
    m = _IMO_PAT.search(last_user or "")
    if m:
      imo = m.group(1)
      ai = AIMessage(
          content="",
          tool_calls=[{
            "id": "auto_get_vessel_data_by_imo",
            "name": "get_vessel_data",
            "args": {"imo": imo}
          }],
      )
      return {"messages": [ai]}

    # 2) deterministic filters
    filters = _extract_asset_filters(last_user or "")
    if filters:
      ai = AIMessage(
          content="",
          tool_calls=[{
            "id": "auto_get_vessel_data_by_filters",
            "name": "get_vessel_data",
            "args": filters
          }],
      )
      return {"messages": [ai]}

    # 3) fallback to LLM for unusual phrasings
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
