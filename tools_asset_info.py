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

# Retry controls
ASSET_MAX_RETRIES = int(os.getenv("ASSET_MAX_RETRIES", "3"))
ASSET_RETRY_BACKOFF = float(os.getenv("ASSET_RETRY_BACKOFF", "0.5"))  # seconds (exponential)


# -----------------------------------------------------------------------------
# Helpers (data extraction)
# -----------------------------------------------------------------------------
def _extract_imo_from_identifiers(identifiers: Any) -> Optional[str]:
  """
  Accept common IMO label variants present in data:
  'IMO', 'IMONUMBER', 'IMO NUMBER', 'IMONBR', 'IMO_NUMBER', etc.
  Returns the IMO value as-is (string), or None.
  """
  def _norm(name: Any) -> str:
    return str(name or "").replace(" ", "").upper()

  acceptable = {"IMO", "IMONUMBER", "IMONBR", "IMO_NUMBER"}
  if isinstance(identifiers, list):
    for ident in identifiers:
      try:
        if _norm((ident or {}).get("name")) in acceptable:
          return (ident or {}).get("value")
      except Exception:
        continue
  elif isinstance(identifiers, dict):
    if _norm((identifiers or {}).get("name")) in acceptable:
      return (identifiers or {}).get("value")
  return None


def _extract_owner(ownership_details: Any) -> Dict[str, Any]:
  """
  ownership_details is nested (list). Pick a sensible owner entry:
  - Prefer company_role like 'Registered Owner' or 'Owner' if present (case-insensitive).
  - Else first item.
  Returns a simple dict {name, company_code, country_* , role}
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

  preferred_roles = {"registered owner", "owner", "commercial owner", "operator", "manager"}
  if isinstance(ownership_details, list) and ownership_details:
    preferred = None
    for e in ownership_details:
      role = ((e or {}).get("company_role") or "").replace("_", " ").lower()
      if role in preferred_roles:
        preferred = e
        break
    return _shape(preferred or ownership_details[0])
  elif isinstance(ownership_details, dict):
    return _shape(ownership_details)
  return {}


# -----------------------------------------------------------------------------
# Query-building utils
# -----------------------------------------------------------------------------
def _one_of(clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Require at least one of the supplied alternatives (mini-OR group)."""
  return {"bool": {"should": clauses, "minimum_should_match": 1}}

def _nested(path: str, query: Dict[str, Any]) -> Dict[str, Any]:
  """Wrap a query under a nested path."""
  return {"nested": {"path": path, "query": query}}

def _keyword_or_text(field_base: str, value: str, phrase: bool = True) -> Dict[str, Any]:
  """
  OR between exact keyword match and analyzed match/phrase on a non-nested field.
  """
  should = [{"term": {f"{field_base}.keyword": value}}]
  if phrase:
    should.append({"match_phrase": {field_base: value}})
  else:
    should.append({"match": {field_base: {"query": value, "operator": "and"}}})
  return _one_of(should)

def _nested_keyword_or_text(path: str, field_base: str, value: str, phrase: bool = True) -> Dict[str, Any]:
  """
  OR between exact keyword match and analyzed match/phrase under a nested path.
  """
  inner = _keyword_or_text(field_base, value, phrase=phrase)
  return _nested(path, inner)

def _numeric_term(field: str, value: Any) -> Optional[Dict[str, Any]]:
  try:
    return {"term": {field: int(value)}}
  except Exception:
    return None


# -----------------------------------------------------------------------------
# Query builder (holistic, mapping-aware)
# -----------------------------------------------------------------------------
def build_asset_query(filters: Dict[str, Any]) -> Dict[str, Any]:
  """
  Build a precise OpenSearch bool query honoring your mappings:

  - Nested fields handled via `nested`:
      * identifiers (IMO/MMSI/name/value)
      * ownership_details (company.name)
  - Non-nested text fields use keyword OR analyzed match/phrase (case-flexible).
  - Numeric fields use range/term in filter (no scoring).
  - Attribute groups (flag/owner/shiptype/name/builder) are REQUIRED (AND),
    each with its own internal OR to support variants.
  """
  must: List[Dict[str, Any]] = []
  filter_: List[Dict[str, Any]] = []

  # ---------- Identifiers (nested) ----------
  # IMO: match regardless of label (IMO/IMONUMBER/etc.) and type (string/number)
  if v := filters.get("imo"):
    imo_should: List[Dict[str, Any]] = [
      {"term": {"identifiers.value.keyword": str(v)}},
      {"match": {"identifiers.value": str(v)}},
    ]
    num_term = _numeric_term("identifiers.value", v)
    if num_term:
      imo_should.append(num_term)
    filter_.append(_nested("identifiers", _one_of(imo_should)))

  # MMSI: match via flag_details.mmsi (long) OR identifiers nested (text/keyword)
  if v := filters.get("mmsi"):
    mmsi_group: List[Dict[str, Any]] = []
    num_term = _numeric_term("flag_details.mmsi", v)
    if num_term:
      mmsi_group.append(num_term)
    # also look in identifiers.value under nested
    id_should = [
      {"term": {"identifiers.value.keyword": str(v)}},
      {"match": {"identifiers.value": str(v)}},
    ]
    id_num_term = _numeric_term("identifiers.value", v)
    if id_num_term:
      id_should.append(id_num_term)
    mmsi_group.append(_nested("identifiers", _one_of(id_should)))
    filter_.append(_one_of(mmsi_group))

  # Name (root 'name' is TEXT ONLY in mapping): use phrase + prefix
  if v := filters.get("name"):
    must.append(_one_of([
      {"match_phrase": {"name": v}},
      {"match_phrase_prefix": {"name": {"query": v, "max_expansions": 50}}},
    ]))

  # ---------- Ranges ----------
  dwt_min = filters.get("dwt_min")
  dwt_max = filters.get("dwt_max")
  if dwt_min is not None or dwt_max is not None:
    r: Dict[str, Any] = {}
    if dwt_min is not None:
      r["gte"] = dwt_min
    if dwt_max is not None:
      r["lte"] = dwt_max
    filter_.append({"range": {"asset_characteristics.deadweight": r}})

  year_min = filters.get("year_min")
  year_max = filters.get("year_max")
  if year_min is not None or year_max is not None:
    r: Dict[str, Any] = {}
    if year_min is not None:
      r["gte"] = year_min
    if year_max is not None:
      r["lte"] = year_max
    filter_.append({"range": {"asset_characteristics.year_of_build": r}})

  # ---------- Attribute groups (REQUIRED) ----------
  # Flag: match flag name or flag country name (text+keyword)
  if v := filters.get("flag"):
    must.append(_one_of([
      _keyword_or_text("flag_details.flag_name", v),
      _keyword_or_text("flag_details.flag_country.country.name", v),
    ]))

  # Owner (nested ownership_details.company.name)
  if v := filters.get("owner"):
    must.append(_nested_keyword_or_text("ownership_details", "ownership_details.company.name", v, phrase=True))

  # Shiptype (text+keyword, analyzer-friendly)
  if v := filters.get("shiptype"):
    must.append(_one_of([
      {"term": {"ship.shiptype_level_5.keyword": v}},
      {"match": {"ship.shiptype_level_5": {"query": v, "operator": "and"}}},
    ]))

  # Builder (text+keyword)
  if v := filters.get("builder"):
    must.append(_keyword_or_text("asset_characteristics.builder", v, phrase=True))

  # Free text over copied fields + name/shiptype/builder (nested copied to full_text_nested)
  if v := filters.get("free_text"):
    must.append({
      "multi_match": {
        "query": v,
        "fields": [
          "name^2",
          "full_text_nested",  # many nested fields are copy_to here
          "ship.shiptype_level_5",
          "asset_characteristics.builder",
        ],
        "type": "best_fields",
        "operator": "and"
      }
    })

  # ---------- Final bool ----------
  bool_body: Dict[str, Any] = {"must": must or [{"match_all": {}}]}
  if filter_:
    bool_body["filter"] = filter_

  # NOTE: Root 'name' has no .keyword in your mapping; keeping name.keyword in sort
  # with unmapped_type='keyword' avoids hard errors and gracefully skips on this index.
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
      # Only original fields belong in _source (no .keyword subfields)
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
        "ownership_details.company._class"
      ]
    }
  }


# -----------------------------------------------------------------------------
# OpenSearch client (with retries & logs)
# -----------------------------------------------------------------------------
def _post_asset(payload: Dict[str, Any]) -> Dict[str, Any]:
  safe_payload = json.dumps(payload, indent=2, ensure_ascii=False)
  logger.info("ASSET POST URL: %s", ASSET_INFO_URL)
  logger.info("ASSET POST BODY:\n%s", safe_payload)

  last_err = None
  for attempt in range(1, ASSET_MAX_RETRIES + 1):
    t0 = time.time()
    try:
      resp = requests.post(
          ASSET_INFO_URL,
          headers={"Content-Type": "application/json", "User-Agent": "gen-ai-asset-info"},
          data=json.dumps(payload),
          auth=(username, password),
          timeout=(5, ASSET_TIMEOUT),  # (connect, read)
          verify=ASSET_VERIFY_TLS,
      )
      dt = (time.time() - t0) * 1000.0
      logger.info("ASSET RESP STATUS: %s (%.1f ms)", resp.status_code, dt)

      # Retry on 429/5xx
      if resp.status_code in (429,) or 500 <= resp.status_code < 600:
        last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:256]}")
        backoff = ASSET_RETRY_BACKOFF * (2 ** (attempt - 1))
        logger.warning("ASSET transient error, retrying in %.2fs (attempt %d/%d)", backoff, attempt, ASSET_MAX_RETRIES)
        time.sleep(backoff)
        continue

      if resp.status_code == 401:
        logger.error("ASSET 401 Unauthorized — check Basic Auth credentials/secret.")
      if resp.status_code >= 400:
        logger.error("ASSET error body (truncated): %s", resp.text[:512])
        resp.raise_for_status()

      try:
        data = resp.json()
      except Exception:
        logger.error("ASSET non-JSON response; content-type=%s body=%s",
                     resp.headers.get("Content-Type"), resp.text[:512])
        raise

      return data

    except requests.RequestException as e:
      last_err = e
      backoff = ASSET_RETRY_BACKOFF * (2 ** (attempt - 1))
      logger.warning("ASSET request exception: %s; retrying in %.2fs (attempt %d/%d)",
                     str(e), backoff, attempt, ASSET_MAX_RETRIES)
      time.sleep(backoff)

  # Out of retries
  logger.error("ASSET request failed after %d attempts: %s", ASSET_MAX_RETRIES, str(last_err))
  raise last_err or Exception("ASSET request failed")


def _count_asset(query: Dict[str, Any]) -> int:
  """
  Send a size=0 version of the query to get total quickly.
  Sort doesn't matter for count; remove it if present.
  """
  count_payload = dict(query)
  count_payload["size"] = 0
  count_payload.pop("sort", None)

  data = _post_asset(count_payload)
  total = (data.get("hits", {}) or {}).get("total")

  # total can be int (legacy) or dict {"value": N, "relation": "eq"/"gte"}
  if isinstance(total, dict):
    val = int(total.get("value") or 0)
    return val
  return int(total or 0)


def _collect_all_hits(
    base_query: Dict[str, Any],
    page_size: Optional[int] = None,
    use_terminate_after: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
  """
  Fetch documents with search_after pagination.
  If use_terminate_after=True, do a single-page early-terminated fetch (sample).
  Returns (hits, reported_total).
  """
  all_hits: List[Dict[str, Any]] = []
  search_after = None
  pages = 0
  total_val = 0
  ps = page_size or base_query.get("size", ASSET_PAGE_SIZE)

  while True:
    pages += 1
    if pages > ASSET_MAX_PAGES:
      logger.warning("ASSET pagination stopped at ASSET_MAX_PAGES=%d", ASSET_MAX_PAGES)
      break

    payload = dict(base_query)
    payload["size"] = ps

    # For small sample requests, we can terminate early per shard
    if use_terminate_after:
      payload["terminate_after"] = ps
      # For faster sampling, you can also disable exact counts
      payload["track_total_hits"] = False

    if search_after:
      payload["search_after"] = search_after

    data = _post_asset(payload)
    hits_list = (data.get("hits", {}) or {}).get("hits", []) or []
    total_obj = (data.get("hits", {}) or {}).get("total")

    if isinstance(total_obj, dict):
      total_val = int(total_obj.get("value") or 0)
    else:
      total_val = int(total_obj or 0)

    logger.info("ASSET page=%d hits_this_page=%d", pages, len(hits_list))
    if not hits_list:
      break

    all_hits.extend(hits_list)

    last_sort = hits_list[-1].get("sort")
    if not last_sort:
      logger.info("ASSET last hit has no 'sort'; stopping pagination.")
      break
    search_after = last_sort

    if len(all_hits) >= total_val:
      break

    # If we only wanted a one-page sample, stop after first page
    if use_terminate_after:
      break

  logger.info("ASSET collected hits=%d (reported total=%d) pages=%d", len(all_hits), total_val, pages)
  return all_hits, int(total_val or 0)


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

    # MMSI may be stored as int; keep as string to preserve leading zeros if any
    raw_mmsi = (src.get("flag_details", {}) or {}).get("mmsi")
    mmsi_str = str(raw_mmsi) if raw_mmsi not in (None, "") else None

    item = {
      "id": src.get("id"),
      "name": src.get("name"),
      "imo": imo_val,
      "mmsi": mmsi_str,
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
  Optimization:
    - If top_n is set and <= page size, fetch a single early-terminated page as a sample.
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
        "Please narrow with additional filters (e.g., shiptype, DWT range, year of build, owner, or name). "
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

    # ---------- Fetch (paged or sampled) ----------
    fetch_page_size = ASSET_PAGE_SIZE
    use_sample_early_terminate = False

    if top_n is not None and top_n > 0:
      fetch_page_size = min(top_n, ASSET_PAGE_SIZE)
      if top_n <= ASSET_PAGE_SIZE:
        use_sample_early_terminate = True

    fetch_query = dict(query)
    fetch_query["size"] = fetch_page_size

    hits, reported_total = _collect_all_hits(
        fetch_query,
        page_size=fetch_page_size,
        use_terminate_after=use_sample_early_terminate,
    )

    if allow_large_result and len(hits) > ASSET_HARD_LIMIT:
      logger.info("ASSET trimming from %d to hard limit %d", len(hits), ASSET_HARD_LIMIT)
      hits = hits[:ASSET_HARD_LIMIT]

    if top_n is not None and top_n > 0 and len(hits) > top_n:
      logger.info("ASSET trimming hits from %d to top_n=%d", len(hits), top_n)
      hits = hits[:top_n]

    # ---------- Normalize ----------
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
    return {
      "type": "vessels",
      "error": f"{type(e).__name__}: {str(e)}",
    }
