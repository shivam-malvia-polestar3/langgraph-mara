import os
import json
import math
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode

# ========= ENV (shared) =========
API_BASE = os.environ.get("API_BASE", "https://account-service-api-public.polestar-production.com")
APPLICATION_API_BASE = os.environ.get("APPLICATION_API_BASE", "https://api.polestar-production.com")
AUTH_URL = f"{API_BASE}/v1/auth/signin"
USER_AGENT = os.environ.get("USER_AGENT", "gen-ai-langgraph")

DEFAULT_PAGE_LIMIT = int(os.environ.get("DEFAULT_PAGE_LIMIT", "200"))

# ========= TIME/FORMAT =========
ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FMT = "%Y-%m-%d"

def utcnow() -> datetime:
  return datetime.utcnow().replace(tzinfo=timezone.utc)

def utcnow_iso() -> str:
  return utcnow().strftime(ISO_FMT)

def get_12_months_old_timestamp() -> str:
  return (datetime.utcnow() - timedelta(days=365)).strftime(ISO_FMT)

def get_current_timestamp() -> str:
  return datetime.utcnow().strftime(ISO_FMT)

def _parse_ts_flexible(value: str) -> datetime:
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

  now = utcnow()
  if dt.date() == now.date():
    adj = now - timedelta(hours=1)
    return adj.strftime(ISO_FMT)
  return dt.strftime(ISO_FMT)

# ========= AUTH (Account Service) =========
_token_cache: Dict[str, any] = {"access_token": None, "expiry": datetime.min}
_creds_cache: Dict[str, str] = {}

# imported by app.py; keep this import path stable
from secret_manager import get_credentials_from_secrets_manager  # noqa: E402

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

# ========= HTTP helpers =========
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

# ========= Normalizers (shared) =========
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

# ========= AIS positions link =========
def fetch_ais_positions_data(mmsi: Optional[str], ts_start: str, ts_end: str) -> Optional[str]:
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
