from typing import Optional
from langchain_core.tools import tool

from tools_common import (
  APPLICATION_API_BASE, _normalize_ts, get_12_months_old_timestamp,
  get_current_timestamp, fetch_all_records,
  normalize_ais_gaps_response, normalize_sts_response,
  normalize_zone_port_events, normalize_spoofing_events,
  construct_vessel_position_url
)

def get_STS_min_duration_hours() -> int: return 6
def get_spoofing_min_duration_hours() -> int: return 72
def get_gaps_min_duration_hours() -> int: return 12

@tool("get_ais_gaps", description="Retrieve normalized AIS gaps for a vessel. Returns type='a_g'.")
def get_ais_gaps(imo: str,
    mmsi: Optional[str] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    min_duration_hours: Optional[int] = None) -> dict:
  """Retrieve normalized AIS gaps for a vessel. Units are HOURS."""
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

  from datetime import datetime, timedelta
  # minus one minute (as in your lambda)
  dt = datetime.strptime(ts_end, "%Y-%m-%dT%H:%M:%SZ") - timedelta(minutes=1)
  ts_end = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

  q = {"timestamp_start": ts_start, "timestamp_end": ts_end}
  if min_duration_hours: q["duration_hours_gte"] = min_duration_hours
  if event_types: q["event_types"] = event_types

  endpoint = f"{APPLICATION_API_BASE}/voyage-insights/v1/vessel-positional-discrepancy/{imo}"
  events, total, urls = fetch_all_records(endpoint, q)
  normalized = normalize_spoofing_events({"data": {"events": events}})
  pos_url = construct_vessel_position_url(mmsi=mmsi, ts_start=ts_start, ts_end=ts_end)
  return {"type": "p_d", "records": normalized, "total": total, "sources": urls, "pos_url": pos_url}
