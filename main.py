import os
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import tools_condition

# ←—— your Secrets Manager helper (file/module name must match)
from secret_manager import get_credentials_from_secrets_manager


# ========= ENV =========
API_BASE = os.environ.get("API_BASE", "https://account-service-api-public.polestar-production.com")
APPLICATION_API_BASE = os.environ.get("APPLICATION_API_BASE", "https://api.polestar-production.com")
AUTH_URL = f"{API_BASE}/v1/auth/signin"
USER_AGENT = os.environ.get("USER_AGENT", "gen-ai-langgraph")
VESSELS_BY_FLAG_ENDPOINT = os.environ.get("VESSELS_BY_FLAG_ENDPOINT")  # optional

# server-side page size for pagination (tweak if backend allows)
DEFAULT_PAGE_LIMIT = int(os.environ.get("DEFAULT_PAGE_LIMIT", "200"))

_token_cache = {"access_token": None, "expiry": datetime.min}
_creds_cache = {}

# ========= IN-MEMORY SESSION STORE (swap to Redis/DB in prod) =========
SESSIONS: Dict[str, List[AnyMessage]] = {}
SESSION_CTX: Dict[str, Dict[str, Optional[str]]] = {}  # per-session defaults: imo/mmsi/timestamp_start/end


# ========= UTIL =========
ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FMT = "%Y-%m-%d"

def get_12_months_old_timestamp() -> str:
  return (datetime.utcnow() - timedelta(days=365)).strftime(ISO_FMT)

def get_current_timestamp() -> str:
  return datetime.utcnow().strftime(ISO_FMT)

def get_STS_min_duration_hours() -> int:
  return 6

def get_spoofing_min_duration_hours() -> int:
  return 72

def get_gaps_min_duration_hours() -> int:
  return 12

def _parse_ts_flexible(value: str) -> datetime:
  """Accept 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ'."""
  try:
    return datetime.strptime(value, ISO_FMT).replace(tzinfo=timezone.utc)
  except ValueError:
    return datetime.strptime(value, DATE_FMT).replace(tzinfo=timezone.utc)

def _normalize_ts(value: Optional[str], role: str) -> str:
  """
  role='start' -> date-only becomes 00:00:00Z
  role='end'   -> date-only becomes 23:59:59Z; if end date is today, use now-1h
  """
  if not value:
    return get_12_months_old_timestamp() if role == "start" else get_current_timestamp()

  dt = _parse_ts_flexible(value)

  if role == "start":
    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.strftime(ISO_FMT)

  # role == 'end'
  is_date_only = len(value) == 10
  if is_date_only:
    dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)

  now = datetime.utcnow().replace(tzinfo=timezone.utc)
  if dt.date() == now.date():
    adj = now - timedelta(hours=1)
    return adj.strftime(ISO_FMT)

  return dt.strftime(ISO_FMT)

def get_cached_access_token() -> str:
  """Return cached bearer token; fetch new if missing/expired (via AWS Secrets Manager)."""
  global _creds_cache
  now = datetime.utcnow()

  if _token_cache["access_token"] and _token_cache["expiry"] > now:
    return _token_cache["access_token"]

  if not _creds_cache:
    _creds_cache = get_credentials_from_secrets_manager()

  username = _creds_cache.get("username")
  password = _creds_cache.get("password")
  if not username or not password:
    raise RuntimeError("Secrets Manager did not return username/password")

  resp = requests.post(
      AUTH_URL,
      headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
      json={"username": username, "password": password},
      timeout=15,
  )
  resp.raise_for_status()
  access_token = resp.headers.get("access-token")
  if not access_token:
    raise ValueError("Auth response missing 'access-token' header")

  _token_cache["access_token"] = access_token
  _token_cache["expiry"] = now + timedelta(minutes=9)
  return access_token

def fetch_data(endpoint: str, query_params: dict) -> dict:
  headers = {"Authorization": f"Bearer {get_cached_access_token()}", "User-Agent": USER_AGENT}
  r = requests.get(endpoint, headers=headers, params=query_params, timeout=30)
  r.raise_for_status()
  data = r.json()
  data["request_url"] = r.url
  return data

def fetch_all_records(endpoint: str, base_query: dict, page_limit: int = DEFAULT_PAGE_LIMIT) -> Tuple[List[dict], int, List[str]]:
  """Iterate over all pages using limit/offset until we've collected all events."""
  all_events: List[dict] = []
  request_urls: List[str] = []
  offset = 0
  total_count: Optional[int] = None

  while True:
    q = {**base_query, "limit": page_limit, "offset": offset}
    res = fetch_data(endpoint, q)
    batch = res.get("data", {}).get("events", []) or []
    request_urls.append(res.get("request_url", ""))

    if total_count is None:
      total_count = res.get("meta", {}).get("total_count")
      if total_count is None:
        total_count = 0  # compute at the end if server doesn't send it

    all_events.extend(batch)
    got = len(batch)

    if got == 0:
      break
    offset += got

    if total_count and offset >= total_count:
      break
    if res.get("meta", {}).get("total_count") is None and got < page_limit:
      break

  if not total_count:
    total_count = len(all_events)
  return all_events, total_count, request_urls


# ========= NORMALIZERS =========
def normalize_sts_response(sts_data: dict) -> List[dict]:
  events = sts_data.get("data", {}).get("events", [])
  out = []
  for e in events:
    pv = e.get("paired_vessel", {})
    loc = e.get("location", {})
    item = {}
    if "latitude" in loc: item["la"] = loc["latitude"]
    if "longitude" in loc: item["lo"] = loc["longitude"]
    if "duration_hours" in e: item["dh"] = e["duration_hours"]
    if "imo" in pv: item["p_imo"] = pv["imo"]
    out.append(item)
  return out

def normalize_ais_gaps_response(ais_data: dict) -> List[dict]:
  events = ais_data.get("data", {}).get("events", [])
  out = []
  for e in events:
    s = e.get("stopped", {})
    r = e.get("resumed", {})
    item = {}
    if "gap_duration_hours" in e: item["dh"] = e["gap_duration_hours"]
    if "latitude" in s: item["s_la"] = s["latitude"]
    if "longitude" in s: item["s_lo"] = s["longitude"]
    if "latitude" in r: item["r_la"] = r["latitude"]
    if "longitude" in r: item["r_lo"] = r["longitude"]
    out.append(item)
  return out

def normalize_zone_port_events(zone_data: dict) -> List[dict]:
  events = zone_data.get("data", {}).get("events", [])
  out = []
  for e in events:
    det = e.get("event_details", {})
    z = e.get("zone_information", {})
    item = {}
    if "event_type" in det: item["et"] = det["event_type"]
    if "name" in z: item["z_n"] = z["name"]
    if "latitude" in det: item["la"] = det["latitude"]
    if "longitude" in det: item["lo"] = det["longitude"]
    out.append(item)
  return out

def normalize_spoofing_events(spoofing_data: dict) -> List[dict]:
  events = spoofing_data.get("data", {}).get("events", [])
  out = []
  for e in events:
    st = e.get("started", {})
    item = {}
    if "latitude" in st: item["la"] = st["latitude"]
    if "longitude" in st: item["lo"] = st["longitude"]
    if "duration_hours" in e: item["dh"] = e["duration_hours"]
    out.append(item)
  return out


# ========= CORE TOOL IMPLEMENTATIONS =========
def construct_vessel_position_url(mmsi: Optional[str], ts_start: str, ts_end: str) -> Optional[str]:
  if not mmsi:
    return None
  ts_start_dt = datetime.strptime(ts_start, ISO_FMT)
  ts_end_dt = datetime.strptime(ts_end, ISO_FMT)
  duration_sec = (ts_end_dt - ts_start_dt).total_seconds()
  duration_days = max(1, (ts_end_dt - ts_start_dt).days)
  multiplier = 4
  raw_count = int(duration_days * (multiplier / (1 + math.log1p(duration_days))))
  position_count = min(50, max(duration_days, raw_count))
  raw_freq = duration_sec / position_count
  downsampling = max(300, math.ceil(raw_freq / 300) * 300)
  endpoint = f"http://internal-stage-aisapi-lb-1254224911.us-east-1.elb.amazonaws.com/api/v2/track/{mmsi}"
  params = {"end_date": ts_end, "position_count": position_count, "downsample_frequency_seconds": downsampling}
  return f"{endpoint}?{urlencode(params)}"


# ---- Tools (LLM-callable) ----
@tool("get_vessel_data", description="Return IMOs for a given flag. Calls VESSELS_BY_FLAG_ENDPOINT if set.")
def get_vessel_data(flag: str) -> dict:
  """Return IMOs for a given flag."""
  if not flag:
    return {"imos": []}
  if VESSELS_BY_FLAG_ENDPOINT:
    try:
      headers = {"Authorization": f"Bearer {get_cached_access_token()}", "User-Agent": USER_AGENT}
      r = requests.get(VESSELS_BY_FLAG_ENDPOINT, headers=headers, params={"flag": flag}, timeout=20)
      r.raise_for_status()
      data = r.json()
      if isinstance(data, dict) and "imos" in data:
        return {"imos": data["imos"]}
    except Exception:
      pass
  return {"imos": []}

@tool("get_ais_gaps", description="Retrieve normalized AIS gaps for a vessel. Returns type='a_g'.")
def get_ais_gaps(imo: str,
    mmsi: Optional[str] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    min_duration_hours: Optional[int] = None) -> dict:
  """Retrieve normalized AIS gaps for a vessel."""
  ts_start = _normalize_ts(timestamp_start, "start")
  ts_end   = _normalize_ts(timestamp_end, "end")
  min_duration_hours = min_duration_hours or get_gaps_min_duration_hours()

  q = {"timestamp_start": ts_start, "timestamp_end": ts_end}
  if min_duration_hours: q["gap_threshold_gte"] = min_duration_hours

  endpoint = f"{APPLICATION_API_BASE}/voyage-insights/v1/vessel-ais-reporting-gaps/{imo}"
  events, total, urls = fetch_all_records(endpoint, q)
  normalized = normalize_ais_gaps_response({"data": {"events": events}})
  pos_url = construct_vessel_position_url(mmsi=mmsi, ts_start=ts_start, ts_end=ts_end)
  return {"type": "a_g", "records": normalized, "total": total, "sources": urls, "pos_url": pos_url}

@tool("get_sts_data", description="Retrieve normalized STS events for a vessel. Returns type='sts'.")
def get_sts_data(imo: str,
    mmsi: Optional[str] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    min_duration_hours: Optional[int] = None,
    sts_type: Optional[str] = None) -> dict:
  """Retrieve normalized STS events for a vessel."""
  ts_start = _normalize_ts(timestamp_start, "start")
  ts_end   = _normalize_ts(timestamp_end, "end")
  min_duration_hours = min_duration_hours or get_STS_min_duration_hours()

  q = {"timestamp_start": ts_start, "timestamp_end": ts_end}
  if sts_type: q["sts_type"] = sts_type
  if min_duration_hours: q["sts_duration_gte"] = min_duration_hours

  endpoint = f"{APPLICATION_API_BASE}/voyage-insights/v1/vessel-sts-pairings/{imo}"
  events, total, urls = fetch_all_records(endpoint, q)
  normalized = normalize_sts_response({"data": {"events": events}})
  pos_url = construct_vessel_position_url(mmsi=mmsi, ts_start=ts_start, ts_end=ts_end)
  return {"type": "sts", "records": normalized, "total": total, "sources": urls, "pos_url": pos_url}

@tool("get_zone_port_events", description="Retrieve normalized zone/port events. Returns type='z_p'.")
def get_zone_port_events(imo: str,
    mmsi: Optional[str] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    event_type: Optional[str] = None) -> dict:
  """Retrieve normalized zone/port events for a vessel."""
  ts_start = _normalize_ts(timestamp_start, "start")
  ts_end   = _normalize_ts(timestamp_end, "end")
  event_type = event_type or "PORT_ARRIVAL,PORT_DEPARTURE"

  q = {"timestamp_start": ts_start, "timestamp_end": ts_end, "event_type": event_type}
  endpoint = f"{APPLICATION_API_BASE}/voyage-insights/v1/vessel-zone-and-port-events/{imo}"
  events, total, urls = fetch_all_records(endpoint, q)
  normalized = normalize_zone_port_events({"data": {"events": events}})
  pos_url = construct_vessel_position_url(mmsi=mmsi, ts_start=ts_start, ts_end=ts_end)
  return {"type": "z_p", "records": normalized, "total": total, "sources": urls, "pos_url": pos_url}

@tool("get_positional_discrepancy", description="Retrieve positional discrepancy (spoofing) events. Returns type='p_d'.")
def get_positional_discrepancy(imo: str,
    mmsi: Optional[str] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    min_duration_hours: Optional[int] = None,
    event_types: Optional[str] = None) -> dict:
  """Retrieve positional discrepancy (spoofing) events for a vessel."""
  ts_start = _normalize_ts(timestamp_start, "start")
  ts_end   = _normalize_ts(timestamp_end, "end")
  min_duration_hours = min_duration_hours or get_spoofing_min_duration_hours()

  # minus one minute as in lambda
  dt = datetime.strptime(ts_end, ISO_FMT) - timedelta(minutes=1)
  ts_end = dt.strftime(ISO_FMT)

  q = {"timestamp_start": ts_start, "timestamp_end": ts_end}
  if min_duration_hours: q["duration_hours_gte"] = min_duration_hours
  if event_types: q["event_types"] = event_types

  endpoint = f"{APPLICATION_API_BASE}/voyage-insights/v1/vessel-positional-discrepancy/{imo}"
  events, total, urls = fetch_all_records(endpoint, q)
  normalized = normalize_spoofing_events({"data": {"events": events}})
  pos_url = construct_vessel_position_url(mmsi=mmsi, ts_start=ts_start, ts_end=ts_end)
  return {"type": "p_d", "records": normalized, "total": total, "sources": urls, "pos_url": pos_url}


TOOLS = [get_vessel_data, get_ais_gaps, get_sts_data, get_zone_port_events, get_positional_discrepancy]
TOOL_MAP = {t.name: t for t in TOOLS}


# ========= LLMs =========
planner = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Per-tool system prompts for summarization (easy to extend)
SUMMARY_SYSTEM_PROMPTS: Dict[str, str] = {
  "sts": "You are an STS analyst. Summarize STS events succinctly in 1 line.",
  "a_g": "You are an AIS-gaps analyst. Summarize gaps: count, typical durations, and notable start/end locations.",
  "z_p": "You analyze zone/port events. Summarize arrivals/departures: top ports/zones and frequency.",
  "p_d": "You analyze positional discrepancies. Summarize likely spoofing: counts, durations, and notable coordinates.",
}


# ========= Parallel tool runner node =========
def _run_tools_concurrently(tool_calls: List[dict]) -> List[ToolMessage]:
  msgs: List[ToolMessage] = []

  def _invoke_one(call):
    name = call.get("name")
    args = call.get("args") or {}
    call_id = call.get("id")
    tool = TOOL_MAP.get(name)
    if not tool:
      return ToolMessage(
          content=json.dumps({"type": name or "unknown", "error": f"Unknown tool {name}"}),
          name=name or "unknown",
          tool_call_id=call_id,
      )
    try:
      result = tool.invoke(args)
    except Exception as e:
      result = {"type": name, "error": f"{type(e).__name__}: {str(e)}"}
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


# ========= Planner node =========
def planner_node(state: MessagesState):
  ai = planner.invoke(state["messages"])
  return {"messages": [ai]}


# ========= Graph =========
builder = StateGraph(MessagesState)
builder.add_node("planner", planner_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", tools_condition)
builder.add_edge("tools", "planner")
graph = builder.compile()


# ========= Per-tool summaries (parallel, outside graph for flexibility) =========
def summarize_per_tool(groups: Dict[str, List[dict]]) -> Dict[str, str]:
  """
  groups: {"sts": [toolReturn, ...], "a_g": [...]}  -> {"sts": "text", "a_g": "text"}
  Run each summary with a **different system instruction** and in **parallel**.
  """
  out: Dict[str, str] = {}

  def _summ(typ: str, payload: List[dict]):
    sys = SUMMARY_SYSTEM_PROMPTS.get(typ, "Summarize succinctly for a maritime analyst.")
    smsg = SystemMessage(content=sys)
    hmsg = HumanMessage(content=f"DATASET_TYPE={typ}\nJSON:\n{json.dumps(payload, ensure_ascii=False)}")
    msg = summarizer.invoke([smsg, hmsg])
    return typ, (msg.content if hasattr(msg, "content") else str(msg))

  # Filter out pure-error payloads for summarization (but keep them in raw_results)
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


# ========= FastAPI models =========
class NLQuery(BaseModel):
  query: str
  session_id: str = Field(description="Required: groups conversation under this session")
  correlation_id: str = Field(description="Required: correlation identifier for matching responses")


# ========= Session helpers =========
def _get_or_create_session(session_id: str) -> str:
  sid = session_id
  SESSIONS.setdefault(sid, [])
  SESSION_CTX.setdefault(sid, {"imo": None, "mmsi": None, "timestamp_start": None, "timestamp_end": None})
  return sid

def _context_system_prompt(sid: str) -> Optional[SystemMessage]:
  ctx = SESSION_CTX.get(sid) or {}
  bits = []
  if ctx.get("imo"):
    bits.append(f"default_imo={ctx['imo']}")
  if ctx.get("mmsi"):
    bits.append(f"default_mmsi={ctx['mmsi']}")
  if ctx.get("timestamp_start") and ctx.get("timestamp_end"):
    bits.append(f"default_time_window={ctx['timestamp_start']}→{ctx['timestamp_end']}")
  if not bits:
    return None
  text = (
      "Conversation context defaults: "
      + ", ".join(bits)
      + ". If the user omits these, use the defaults when calling tools."
  )
  return SystemMessage(content=text)

def _update_ctx_from_tool_calls(sid: str, messages: List[AnyMessage]):
  """After a run, capture last used imo/mmsi/timestamps from tool call args and store as session defaults."""
  ctx = SESSION_CTX.setdefault(sid, {})
  for m in messages:
    if isinstance(m, AIMessage):
      tcs = getattr(m, "tool_calls", None) or []
      for c in tcs:
        args = c.get("args") or {}
        if "imo" in args and args["imo"]:
          ctx["imo"] = str(args["imo"])
        if "mmsi" in args and args["mmsi"]:
          ctx["mmsi"] = str(args["mmsi"])
        if "timestamp_start" in args and args["timestamp_start"]:
          ctx["timestamp_start"] = _normalize_ts(args["timestamp_start"], "start")
        if "timestamp_end" in args and args["timestamp_end"]:
          ctx["timestamp_end"] = _normalize_ts(args["timestamp_end"], "end")


# ========= FastAPI =========
app = FastAPI(title="Polestar LangGraph (Parallel tools + per-tool summaries + session memory)")

@app.post("/run-graph")
def run_graph(body: NLQuery):
  # session + correlation (both required by schema)
  session_id = _get_or_create_session(body.session_id)
  correlation_id = body.correlation_id

  # build message list: persistent history + system instructions
  history = SESSIONS[session_id][:]
  base_system = SystemMessage(content=(
    "You are a maritime analytics assistant. "
    "If a user asks by flag, call get_vessel_data(flag) to obtain IMOs. "
    "Then call the relevant per-IMO tools (e.g., get_sts_data, get_ais_gaps). "
    "If IMO/MMSI/time range are not specified in the new query, use the session defaults. "
    "If an MMSI is provided, include it in tool args to populate pos_url. "
    "You may issue multiple tool calls. Prefer to make all necessary tool calls before replying."
  ))
  ctx_system = _context_system_prompt(session_id)
  msgs: List[AnyMessage] = []
  msgs.append(base_system)
  if ctx_system: msgs.append(ctx_system)
  msgs.extend(history)
  msgs.append(HumanMessage(body.query))

  # run graph
  final = graph.invoke({"messages": msgs})

  # collect tool outputs
  tool_results: List[dict] = []
  for m in final["messages"]:
    if isinstance(m, ToolMessage):
      try:
        tool_results.append(json.loads(m.content))
      except Exception:
        pass

  # group by 'type'
  grouped: Dict[str, List[dict]] = {}
  for r in tool_results:
    typ = r.get("type", "unknown")
    grouped.setdefault(typ, []).append(r)

  # per-type summaries
  summaries = summarize_per_tool(grouped) if grouped else {}

  # last AI text
  final_ai = next((m for m in reversed(final["messages"]) if getattr(m, "type", "") == "ai"), None)
  final_text = getattr(final_ai, "content", "") if final_ai else ""

  # ===== persist conversation =====
  SESSIONS[session_id].append(HumanMessage(body.query))
  new_tail = final["messages"][-3:] if len(final["messages"]) >= 3 else final["messages"]
  for m in new_tail:
    if isinstance(m, (AIMessage, ToolMessage)):
      SESSIONS[session_id].append(m)

  # update session defaults from tool call args
  _update_ctx_from_tool_calls(session_id, final["messages"])

  return {
    "session_id": session_id,
    "correlation_id": correlation_id,
    "nl_final_text": final_text,
    "summaries": summaries,
    "raw_results": grouped,
  }


@app.get("/sessions/{session_id}/summary")
def summarize_session(session_id: str):
  """Summarize the entire conversation history for a session."""
  if session_id not in SESSIONS or not SESSIONS[session_id]:
    return {"session_id": session_id, "summary": ""}

  convo = SESSIONS[session_id]
  # Compact transcript
  lines = []
  for m in convo:
    t = getattr(m, "type", "")
    c = getattr(m, "content", "")
    if isinstance(m, ToolMessage):
      try:
        payload = json.loads(c)
        ttype = payload.get("type", "tool")
        lines.append(f"[tool:{ttype}]")
      except Exception:
        lines.append("[tool]")
    elif t == "ai":
      lines.append(f"AI: {c}")
    elif t == "human":
      lines.append(f"User: {c}")
  transcript = "\n".join(lines)

  sys = SystemMessage(content="Summarize the following conversation for a maritime analytics context, capturing intent, key tools/data retrieved, and outcomes.")
  h = HumanMessage(content=transcript)
  msg = summarizer.invoke([sys, h])
  return {"session_id": session_id, "summary": getattr(msg, "content", "")}


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
  """Clear a session's history and context defaults."""
  SESSIONS.pop(session_id, None)
  SESSION_CTX.pop(session_id, None)
  return {"session_id": session_id, "cleared": True}


@app.get("/healthz")
def healthz():
  return {"ok": True}
