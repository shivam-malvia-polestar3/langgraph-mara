import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import requests
from langchain_core.tools import tool
from secret_manager import get_basic_auth_credentials_from_secrets_manager

# -----------------------------------------------------------------------------
# Basic Auth (Asset Info)
# -----------------------------------------------------------------------------
username, password = get_basic_auth_credentials_from_secrets_manager()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("ASSET_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
logger = logging.getLogger("asset_info")
if not logger.handlers:
  h = logging.StreamHandler()
  fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [asset_info] %(message)s")
  h.setFormatter(fmt)
  logger.addHandler(h)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ASSET_INFO_BASE = os.environ.get("ASSET_INFO_BASE", "https://asset-info-opens.polestar-testing.com")
ASSET_INDEX_PATH = os.environ.get("ASSET_INDEX_PATH", "/asset/_search")
ASSET_INFO_URL = f"{ASSET_INFO_BASE.rstrip('/')}{ASSET_INDEX_PATH}"

ASSET_PAGE_SIZE = int(os.getenv("ASSET_PAGE_SIZE", "200"))
ASSET_MAX_PAGES = int(os.getenv("ASSET_MAX_PAGES", "50"))  # safety guard
ASSET_TIMEOUT = int(os.getenv("ASSET_TIMEOUT", "30"))
ASSET_VERIFY_TLS = os.getenv("ASSET_VERIFY_TLS", "true").lower() != "false"

# Fanout controls
ASSET_NARROW_THRESHOLD = int(os.getenv("ASSET_NARROW_THRESHOLD", "100"))  # ask user to narrow if above this
ASSET_HARD_LIMIT = int(os.getenv("ASSET_HARD_LIMIT", "100"))  # absolute fetch cap when allow_large_result=True


# -----------------------------------------------------------------------------
# Query building
# -----------------------------------------------------------------------------
def _term_or_match(field: str, value: str) -> Dict[str, Any]:
  """Pick keyword term for structured fields, fallback to match for text."""
  keyword_candidates = {
    "flag_details.flag_name",
    "flag_details.flag_country.country.name",
    "name",
    "ship.shiptype_level_5",
    "ownership_details.company.name",
    "asset_characteristics.builder",
  }
  if field in keyword_candidates:
    return {"term": {f"{field}.keyword": value}}
  return {"match": {field: value}}

def _range(field: str, gte: Optional[Any] = None, lte: Optional[Any] = None) -> Dict[str, Any]:
  r: Dict[str, Any] = {}
  if gte is not None:
    r["gte"] = gte
  if lte is not None:
    r["lte"] = lte
  return {"range": {field: r}}

def build_asset_query(filters: Dict[str, Any]) -> Dict[str, Any]:
  """
  Build a tight OpenSearch bool query from normalized filters.
  """
  must: List[Dict[str, Any]] = []
  should: List[Dict[str, Any]] = []
  filter_: List[Dict[str, Any]] = []

  # Structured identifiers first (most selective)
  if v := filters.get("imo"):
    must.append({"term": {"identifiers.value.keyword": str(v)}})
    must.append({"match": {"identifiers.name": "IMO"}})

  if v := filters.get("mmsi"):
    try:
      mmsi_num = int(v)
      filter_.append({"term": {"flag_details.mmsi": mmsi_num}})
    except Exception:
      must.append({"match": {"flag_details.mmsi": str(v)}})

  if v := filters.get("name"):
    should.append(_term_or_match("name", v))

  if v := filters.get("flag"):
    should.append(_term_or_match("flag_details.flag_name", v))
    should.append(_term_or_match("flag_details.flag_country.country.name", v))

  if v := filters.get("owner"):
    should.append(_term_or_match("ownership_details.company.name", v))

  if v := filters.get("builder"):
    should.append(_term_or_match("asset_characteristics.builder", v))

  if v := filters.get("shiptype"):
    should.append(_term_or_match("ship.shiptype_level_5", v))

  dwt_min = filters.get("dwt_min")
  dwt_max = filters.get("dwt_max")
  if dwt_min is not None or dwt_max is not None:
    filter_.append(_range("asset_characteristics.deadweight", dwt_min, dwt_max))

  year_min = filters.get("year_min")
  year_max = filters.get("year_max")
  if year_min is not None or year_max is not None:
    filter_.append(_range("asset_characteristics.year_of_build", year_min, year_max))

  if v := filters.get("free_text"):
    should.append({
      "multi_match": {
        "query": v,
        "fields": [
          "name^2",
          "full_text_nested",
          "ship.shiptype_level_5",
          "ownership_details.company.name",
          "asset_characteristics.builder"
        ],
        "type": "best_fields",
        "operator": "and"
      }
    })

  bool_body: Dict[str, Any] = {"must": must or [{"match_all": {}}]}
  if filter_:
    bool_body["filter"] = filter_
  if should:
    bool_body["should"] = should
    bool_body["minimum_should_match"] = 1

  sort = [
    {"asset_characteristics.year_of_build": {"order": "desc", "unmapped_type": "long"}},
    {"name.keyword": {"order": "asc", "unmapped_type": "keyword"}},
    {"_id": {"order": "asc"}},
  ]

  return {
    "size": ASSET_PAGE_SIZE,
    "track_total_hits": True,
    "query": {"bool": bool_body},
    "sort": sort,
    "_source": {
      "includes": [
        "id",
        "name",
        "identifiers",
        "flag_details.flag_name",
        "flag_details.flag_country.country.name",
        "flag_details.mmsi",
        "asset_characteristics.*",
        "ship.shiptype_level_5",
        "ownership_details.company.name",
        "ownership_details.company.country_of_registration.name",
        "ownership_details.company.country_of_domicile.name",
        "ownership_details.company.country_of_control.name",
        "ownership_details.company.company_code",
        "ownership_details.company._class",
        "ownership_details.company.name.keyword"
      ]
    }
  }

# -----------------------------------------------------------------------------
# OpenSearch client (with logs)
# -----------------------------------------------------------------------------
def _post_asset(payload: Dict[str, Any]) -> Dict[str, Any]:
  safe_payload = json.dumps(payload, indent=2)
  logger.info("ASSET POST URL: %s", ASSET_INFO_URL)
  logger.info("ASSET POST BODY:\n%s", safe_payload)

  t0 = time.time()
  try:
    resp = requests.post(
        ASSET_INFO_URL,
        headers={"Content-Type": "application/json", "User-Agent": "gen-ai-asset-info"},
        data=json.dumps(payload),
        auth=(username, password),
        timeout=ASSET_TIMEOUT,
        verify=ASSET_VERIFY_TLS,
    )
  except requests.RequestException as e:
    logger.error("ASSET request exception: %s", str(e))
    raise

  dt = (time.time() - t0) * 1000.0
  logger.info("ASSET RESP STATUS: %s (%.1f ms)", resp.status_code, dt)

  if resp.status_code == 401:
    logger.error("ASSET 401 Unauthorized — check Basic Auth credentials/secret.")
  if resp.status_code >= 400:
    logger.error("ASSET error body (truncated): %s", resp.text[:512])
    resp.raise_for_status()

  try:
    data = resp.json()
  except Exception:
    logger.error("ASSET non-JSON response: %s", resp.text[:512])
    raise
  return data

def _count_asset(query: Dict[str, Any]) -> int:
  """Send a size=0 version of the query to get total quickly."""
  count_payload = dict(query)
  count_payload["size"] = 0
  data = _post_asset(count_payload)
  total = data.get("hits", {}).get("total")
  total_val = total.get("value") if isinstance(total, dict) else total or 0
  logger.info("ASSET preflight total=%s", total_val)
  return int(total_val)

def _collect_all_hits(query: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
  all_hits: List[Dict[str, Any]] = []
  search_after = None
  pages = 0
  total_val = 0

  while True:
    pages += 1
    if pages > ASSET_MAX_PAGES:
      logger.warning("ASSET pagination stopped at ASSET_MAX_PAGES=%d", ASSET_MAX_PAGES)
      break

    payload = dict(query)
    if search_after:
      payload["search_after"] = search_after

    data = _post_asset(payload)
    hits = data.get("hits", {}).get("hits", []) or []
    total = data.get("hits", {}).get("total")
    total_val = total.get("value") if isinstance(total, dict) else (total or 0)

    logger.info("ASSET page=%d hits_this_page=%d", pages, len(hits))
    if not hits:
      break

    all_hits.extend(hits)

    last_sort = hits[-1].get("sort")
    if not last_sort:
      logger.info("ASSET last hit has no 'sort'; stopping pagination.")
      break
    search_after = last_sort

    if len(all_hits) >= total_val:
      break

  logger.info("ASSET collected hits=%d (reported total=%d) pages=%d", len(all_hits), total_val, pages)
  return all_hits, int(total_val or 0)

# -----------------------------------------------------------------------------
# Helpers for nested structures
# -----------------------------------------------------------------------------
def _extract_imo_from_identifiers(identifiers: Any) -> Optional[str]:
  """identifiers is nested; handle list safely."""
  if isinstance(identifiers, list):
    for ident in identifiers:
      try:
        if (ident or {}).get("name", "").upper() == "IMO":
          return (ident or {}).get("value")
      except Exception:
        continue
  elif isinstance(identifiers, dict):
    # Some odd doc? try dict form
    if (identifiers or {}).get("name", "").upper() == "IMO":
      return (identifiers or {}).get("value")
  return None

def _extract_owner(ownership_details: Any) -> Dict[str, Any]:
  """
  ownership_details is nested (list). Pick a sensible owner entry:
  - Prefer company_role like 'Registered Owner' or 'Owner' if present.
  - Else first item.
  Returns a simple dict {name, company_code, country_*}
  """
  def _shape(entry: Dict[str, Any]) -> Dict[str, Any]:
    comp = (entry or {}).get("company", {}) or {}
    return {
      "name": comp.get("name"),
      "company_code": comp.get("company_code"),
      "country_of_registration": (comp.get("country_of_registration", {}) or {}).get("name"),
      "country_of_domicile": (comp.get("country_of_domicile", {}) or {}).get("name"),
      "country_of_control": (comp.get("country_of_control", {}) or {}).get("name"),
      "role": (entry or {}).get("company_role"),
    }

  if isinstance(ownership_details, list) and ownership_details:
    # try to find a preferred role
    preferred = None
    for e in ownership_details:
      role = (e or {}).get("company_role", "") or ""
      if role.lower() in {"registered owner", "owner", "commercial owner", "operator", "manager"}:
        preferred = e
        break
    return _shape(preferred or ownership_details[0])
  elif isinstance(ownership_details, dict):
    return _shape(ownership_details)
  return {}

# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
def _normalize_vessel_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
  try:
    src = doc.get("_source", {}) if isinstance(doc, dict) else {}
    if not isinstance(src, dict):
      logger.warning("ASSET hit with unexpected _source type: %s", type(src))
      return {}

    identifiers = src.get("identifiers", [])
    ownership_details = src.get("ownership_details", [])

    imo_val = _extract_imo_from_identifiers(identifiers)
    owner_val = _extract_owner(ownership_details)

    item = {
      "id": src.get("id"),
      "name": src.get("name"),
      "imo": imo_val,
      "mmsi": (src.get("flag_details", {}) or {}).get("mmsi"),
      "flag": (src.get("flag_details", {}) or {}).get("flag_name"),
      "flag_country": (
        (src.get("flag_details", {}) or {})
        .get("flag_country", {})
        .get("country", {})
        .get("name")
      ),
      "shiptype": (src.get("ship", {}) or {}).get("shiptype_level_5"),
      "dwt": (src.get("asset_characteristics", {}) or {}).get("deadweight"),
      "year_of_build": (src.get("asset_characteristics", {}) or {}).get("year_of_build"),
      "length_overall": (src.get("asset_characteristics", {}) or {}).get("length_overall"),
      "breadth": (src.get("asset_characteristics", {}) or {}).get("breadth"),
      "builder": (src.get("asset_characteristics", {}) or {}).get("builder"),
      "owner": owner_val,
    }
    return item
  except Exception as e:
    logger.exception("ASSET normalize error on doc: %s", str(e))
    return {}

# -----------------------------------------------------------------------------
# LangChain tool
# -----------------------------------------------------------------------------
@tool(
    "get_vessel_data",
    description=(
        "Search Asset Information (OpenSearch) for vessel details. "
        "Accepts filters like imo, mmsi, name, flag, owner, builder, shiptype, "
        "dwt_min, dwt_max, year_min, year_max, free_text. Returns type='vessels'. "
        "Includes fanout controls to avoid huge result sets."
    ),
)
def get_vessel_data(
    imo: Optional[str] = None,
    mmsi: Optional[str] = None,
    name: Optional[str] = None,
    flag: Optional[str] = None,
    owner: Optional[str] = None,
    builder: Optional[str] = None,
    shiptype: Optional[str] = None,
    dwt_min: Optional[int] = None,
    dwt_max: Optional[int] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    free_text: Optional[str] = None,
    top_n: Optional[int] = None,
    allow_large_result: Optional[bool] = False,
) -> Dict[str, Any]:
  """
  Build an OpenSearch query from provided filters. Preflight count first:
    - If total > ASSET_NARROW_THRESHOLD and allow_large_result is False:
        return requires_narrowing=True with no data.
    - Else page through results (capped by ASSET_MAX_PAGES / ASSET_HARD_LIMIT).
  """
  try:
    filters: Dict[str, Any] = {
      "imo": imo,
      "mmsi": mmsi,
      "name": name,
      "flag": flag,
      "owner": owner,
      "builder": builder,
      "shiptype": shiptype,
      "dwt_min": dwt_min,
      "dwt_max": dwt_max,
      "year_min": year_min,
      "year_max": year_max,
      "free_text": free_text,
    }
    filters = {k: v for k, v in filters.items() if v not in (None, "", [])}

    logger.info("ASSET filters parsed: %s", json.dumps(filters, ensure_ascii=False))
    query = build_asset_query(filters)

    # ---------- Preflight count ----------
    total = _count_asset(query)

    # If too large, ask user to narrow unless explicitly allowed
    if not allow_large_result and total > ASSET_NARROW_THRESHOLD:
      message = (
        f"Your query would return {total} vessels, which is too many to fetch. "
        "Please narrow with additional filters (e.g., shiptype, DWT range, year of build, owner, or name prefix). "
        "You can also specify 'top N' (e.g., top 50) if you want a small sample."
      )
      logger.warning(
          "ASSET refusing fanout: total=%d > threshold=%d (set allow_large_result=true to override)",
          total, ASSET_NARROW_THRESHOLD,
      )
      return {
        "type": "vessels",
        "records": [],
        "total": total,
        "sources": [ASSET_INFO_URL],
        "requires_narrowing": True,
        "message": message,
      }

    # If allowed, still hard-cap
    if allow_large_result and total > ASSET_HARD_LIMIT:
      logger.warning("ASSET hard-capping large result: total=%d > hard_limit=%d", total, ASSET_HARD_LIMIT)

    # ---------- Fetch (paged) ----------
    hits, reported_total = _collect_all_hits(query)

    # Apply hard cap if needed
    if allow_large_result and len(hits) > ASSET_HARD_LIMIT:
      logger.info("ASSET trimming from %d to hard limit %d", len(hits), ASSET_HARD_LIMIT)
      hits = hits[:ASSET_HARD_LIMIT]

    # Optional sample trimming
    if top_n is not None and top_n > 0 and len(hits) > top_n:
      logger.info("ASSET trimming hits from %d to top_n=%d", len(hits), top_n)
      hits = hits[:top_n]

    logger.info("ASSET normalizing %d hits", len(hits))
    vessels: List[Dict[str, Any]] = []
    bad = 0
    for h in hits:
      norm = _normalize_vessel_doc(h)
      if norm:
        vessels.append(norm)
      else:
        bad += 1
    if bad:
      logger.warning("ASSET skipped %d malformed hits during normalization", bad)

    return {
      "type": "vessels",
      "records": vessels,
      "total": reported_total,
      "sources": [ASSET_INFO_URL],
      "requires_narrowing": False,
    }

  except Exception as e:
    logger.exception("ASSET tool error: %s", str(e))
    # Return error in-band so the planner/tool runner can surface it
    return {
      "type": "vessels",
      "error": f"{type(e).__name__}: {str(e)}",
    }
