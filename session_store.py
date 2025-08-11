import re
from typing import Dict, List, Optional
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage

from tools_common import utcnow_iso, _normalize_ts

SESSIONS: Dict[str, List[AnyMessage]] = {}
SESSION_CTX: Dict[str, Dict[str, Optional[str]]] = {}
SESSIONS_META: Dict[str, Dict[str, Optional[str]]] = {}

def get_or_create_session(session_id: str) -> str:
  SESSIONS.setdefault(session_id, [])
  SESSION_CTX.setdefault(session_id, {
    "imo": None, "mmsi": None,
    "timestamp_start": None, "timestamp_end": None,
    "last_vessels": None, "selection_limit": None,
    "min_duration_hours_hint": None
  })
  SESSIONS_META.setdefault(session_id, {
    "created_at": utcnow_iso(), "last_activity_at": utcnow_iso(), "first_user_message": None
  })
  return session_id

def context_system_prompt(session_id: str) -> Optional[SystemMessage]:
  ctx = SESSION_CTX.get(session_id) or {}
  bits = []
  if ctx.get("imo"): bits.append(f"default_imo={ctx['imo']}")
  if ctx.get("mmsi"): bits.append(f"default_mmsi={ctx['mmsi']}")
  if ctx.get("timestamp_start") and ctx.get("timestamp_end"):
    bits.append(f"default_time_window={ctx['timestamp_start']}→{ctx['timestamp_end']}")
  if ctx.get("last_vessels"):
    n = len(ctx["last_vessels"])
    if n <= 15:
      bits.append(f"last_vessels={','.join([str(x) for x in ctx['last_vessels']])}")
    bits.append(f"last_vessels_available={n}")
  if ctx.get("selection_limit"):
    bits.append(f"selection_limit={ctx['selection_limit']}")
  if ctx.get("min_duration_hours_hint"):
    bits.append(f"min_duration_hours_hint={ctx['min_duration_hours_hint']}")
  if not bits:
    return None
  txt = (
      "Conversation context defaults and hints: " + ", ".join(bits) + ". "
                                                                      "If the user omits these, use the defaults when calling tools. "
                                                                      "If essential items (IMO or last_vessels for per-vessel tools) are missing, ask a clarifying question."
  )
  return SystemMessage(content=txt)

_MONTHS = {
  "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
  "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
  "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
}

def _try_make_date(month_name: str, year: str, role: str) -> Optional[str]:
  from datetime import datetime, timedelta, timezone
  m = _MONTHS.get(month_name.lower())
  if not m: return None
  if role == "start":
    dt = datetime(int(year), m, 1, tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")
  if m == 12:
    last = datetime(int(year)+1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
  else:
    last = datetime(int(year), m+1, 1, tzinfo=timezone.utc) - timedelta(days=1)
  return last.strftime("%Y-%m-%d")

# robust duration seeding: accepts >=, >, greater than/then, more than, over, longer than
_DURATION_SEED = re.compile(
    r'(?:duration|gap(?:\s*duration)?)\s*(?:>=|>|at\s*least|more\s+than|greater\s+(?:than|then)|over|longer\s+than)\s*(\d+)\s*(?:h|hr|hrs|hour|hours)\b',
    re.IGNORECASE
)

def seed_defaults_from_query(session_id: str, query: str):
  ctx = SESSION_CTX.setdefault(session_id, {})

  m_imo = re.search(r'\bIMO\s*([0-9]{7})\b', query, re.IGNORECASE) or re.search(r'\bwith\s+imo\s+([0-9]{7})\b', query, re.IGNORECASE)
  if m_imo: ctx["imo"] = m_imo.group(1)
  m_mmsi = re.search(r'\bMMSI\s*([0-9]{9})\b', query, re.IGNORECASE)
  if m_mmsi: ctx["mmsi"] = m_mmsi.group(1)

  m_top = re.search(r'\btop\s+(\d+)\b', query, re.IGNORECASE)
  if m_top: ctx["selection_limit"] = m_top.group(1)

  m_dur = _DURATION_SEED.search(query or "")
  if m_dur:
    ctx["min_duration_hours_hint"] = m_dur.group(1)

  m_range = re.search(r'\b([A-Za-z]+)\s+(\d{4})\s*(?:to|-|–|—)\s*([A-Za-z]+)\s+(\d{4})\b', query, re.IGNORECASE)
  if m_range:
    s_mon, s_year, e_mon, e_year = m_range.groups()
    s_raw = _try_make_date(s_mon, s_year, "start")
    e_raw = _try_make_date(e_mon, e_year, "end")
    if s_raw: ctx["timestamp_start"] = _normalize_ts(s_raw, "start")
    if e_raw: ctx["timestamp_end"]   = _normalize_ts(e_raw, "end")
    return

  m_single = re.search(r'\b([A-Za-z]+)\s+(\d{4})\b', query, re.IGNORECASE)
  if m_single:
    mon, yr = m_single.groups()
    s_raw = _try_make_date(mon, yr, "start")
    e_raw = _try_make_date(mon, yr, "end")
    if s_raw and e_raw:
      ctx["timestamp_start"] = _normalize_ts(s_raw, "start")
      ctx["timestamp_end"]   = _normalize_ts(e_raw, "end")

def update_ctx_from_tool_calls(session_id: str, new_messages: List[AnyMessage]):
  ctx = SESSION_CTX.setdefault(session_id, {})
  for m in new_messages:
    if isinstance(m, AIMessage):
      tcs = getattr(m, "tool_calls", None) or []
      for c in tcs:
        args = c.get("args") or {}
        if args.get("imo"): ctx["imo"] = str(args["imo"])
        if args.get("mmsi"): ctx["mmsi"] = str(args["mmsi"])
        if args.get("timestamp_start"): ctx["timestamp_start"] = _normalize_ts(args["timestamp_start"], "start")
        if args.get("timestamp_end"):   ctx["timestamp_end"]   = _normalize_ts(args["timestamp_end"], "end")
        if args.get("min_duration_hours"): ctx["min_duration_hours_hint"] = str(args["min_duration_hours"])
  import json
  for m in new_messages:
    if isinstance(m, ToolMessage):
      try:
        payload = json.loads(m.content)
      except Exception:
        continue
      if isinstance(payload, dict) and payload.get("type") == "vessels":
        v = payload.get("imos") or []
        if v:
          try:
            lim = int(ctx.get("selection_limit") or 0)
          except Exception:
            lim = 0
          if lim and lim > 0:
            v = v[:lim]
          ctx["last_vessels"] = v
