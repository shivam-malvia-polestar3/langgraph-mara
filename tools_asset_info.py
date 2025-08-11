import os
import json
import requests
from typing import Optional, Dict, List, Tuple

from langchain_core.tools import tool

from tools_common import USER_AGENT
# OpenSearch URL + optional auth for the Asset Information Service
ASSET_INFO_URL = os.environ.get(
    "ASSET_INFO_URL",
    "https://asset-info-opens.polestar-testing.com/asset/_search"
)
ASSET_INFO_BEARER = os.environ.get("ASSET_INFO_BEARER")
ASSET_INFO_BASIC_USER = "admin"
ASSET_INFO_BASIC_PASS = ",%jQdbX$D3{@7XWx"

def _asset_info_headers() -> dict:
  headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
  if ASSET_INFO_BEARER:
    headers["Authorization"] = f"Bearer {ASSET_INFO_BEARER}"
  return headers

def _asset_info_auth():
  if ASSET_INFO_BASIC_USER and ASSET_INFO_BASIC_PASS:
    return (ASSET_INFO_BASIC_USER, ASSET_INFO_BASIC_PASS)
  return None

def _build_asset_info_query(
    free_text: Optional[str] = None,
    flag: Optional[str] = None,
    imo: Optional[str] = None,
    mmsi: Optional[str] = None,
    name: Optional[str] = None,
    call_sign: Optional[str] = None,
    shiptype: Optional[str] = None,
    dwt_min: Optional[int] = None,
    dwt_max: Optional[int] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    builder: Optional[str] = None,
    owner: Optional[str] = None,
    country_of_build: Optional[str] = None,
    extra_terms: Optional[Dict[str, str]] = None,
) -> dict:
  must, should, filter_clauses = [], [], []
  query: Dict[str, any] = {
    "track_total_hits": True,
    "_source": {
      "includes": [
        "id",
        "name",
        "flag_details.flag_name",
        "flag_details.flag_country.country.name",
        "flag_details.mmsi",
        "flag_details.call_sign",
        "identifiers.name",
        "identifiers.value",
        "asset_characteristics.deadweight",
        "asset_characteristics.year_of_build",
        "asset_characteristics.builder",
        "ship.shiptype_level_5",
        "ship.ship_status",
        "ownership_details.company.name",
      ]
    },
    "query": {"bool": {"must": must, "should": should, "filter": filter_clauses}}
  }

  if free_text:
    should.append({
      "multi_match": {
        "query": free_text,
        "type": "best_fields",
        "fields": [
          "name^4",
          "full_text_nested^3",
          "identifiers.value^3",
          "flag_details.flag_name^3",
          "flag_details.flag_country.country.name^3",
          "ship.shiptype_level_5^2",
          "ownership_details.company.name^2",
          "asset_characteristics.builder",
        ],
        "operator": "and"
      }
    })

  if flag:
    must.append({
      "bool": {
        "should": [
          {"term": {"flag_details.flag_name.keyword": flag}},
          {"term": {"flag_details.flag_country.country.name.keyword": flag}},
        ],
        "minimum_should_match": 1
      }
    })

  if imo:
    must.append({
      "nested": {
        "path": "identifiers",
        "query": {
          "bool": {
            "must": [
              {"term": {"identifiers.name.keyword": "imo"}},
              {"term": {"identifiers.value.keyword": str(imo)}}
            ]
          }
        }
      }
    })

  if mmsi:
    must.append({"term": {"flag_details.mmsi": int(mmsi)}})

  if name:
    should.append({
      "multi_match": {
        "query": name,
        "fields": ["name^4", "full_text_nested^2"],
        "operator": "and"
      }
    })

  if call_sign:
    must.append({"term": {"flag_details.call_sign.keyword": call_sign}})

  if shiptype:
    should.append({"match_phrase": {"ship.shiptype_level_5": {"query": shiptype}}})

  if builder:
    should.append({
      "multi_match": {
        "query": builder,
        "fields": ["asset_characteristics.builder^3", "full_text_nested"],
        "operator": "and"
      }
    })

  if owner:
    should.append({
      "multi_match": {
        "query": owner,
        "fields": ["ownership_details.company.name^3", "full_text_nested"],
        "operator": "and"
      }
    })

  if country_of_build:
    should.append({
      "multi_match": {
        "query": country_of_build,
        "fields": [
          "asset_characteristics.country_of_build.name",
          "asset_characteristics.country_of_build.alt_name",
          "asset_characteristics.country_of_build.iso2code",
          "asset_characteristics.country_of_build.iso3code",
        ],
        "operator": "or"
      }
    })

  if dwt_min is not None or dwt_max is not None:
    rng: Dict[str, any] = {}
    if dwt_min is not None: rng["gte"] = int(dwt_min)
    if dwt_max is not None: rng["lte"] = int(dwt_max)
    filter_clauses.append({"range": {"asset_characteristics.deadweight": rng}})

  if year_min is not None or year_max is not None:
    rng: Dict[str, any] = {}
    if year_min is not None: rng["gte"] = int(year_min)
    if year_max is not None: rng["lte"] = int(year_max)
    filter_clauses.append({"range": {"asset_characteristics.year_of_build": rng}})

  if extra_terms:
    for field, value in extra_terms.items():
      filter_clauses.append({"term": {field: value}})

  if should:
    query["query"]["bool"]["minimum_should_match"] = 1
  return query

def _normalize_asset_hit(hit: dict) -> dict:
  src = hit.get("_source", {})
  imo_value = None
  for idf in (src.get("identifiers") or []):
    if (idf.get("name") or "").lower() == "imo":
      imo_value = idf.get("value"); break

  owners = []
  for od in (src.get("ownership_details") or []):
    comp = od.get("company") or {}
    nm = comp.get("name")
    if nm and nm not in owners:
      owners.append(nm)

  flag_name = (src.get("flag_details") or {}).get("flag_name")
  flag_country = ((src.get("flag_details") or {}).get("flag_country") or {}).get("country") or {}
  flag_country_name = flag_country.get("name")
  mmsi_value = (src.get("flag_details") or {}).get("mmsi")

  return {
    "id": src.get("id"),
    "name": src.get("name"),
    "imo": imo_value,
    "mmsi": mmsi_value,
    "flag": flag_name or flag_country_name,
    "deadweight": ((src.get("asset_characteristics") or {}).get("deadweight")),
    "year_of_build": ((src.get("asset_characteristics") or {}).get("year_of_build")),
    "builder": ((src.get("asset_characteristics") or {}).get("builder")),
    "shiptype": ((src.get("ship") or {}).get("shiptype_level_5")),
    "owner_names": owners,
    "raw": src
  }

def _asset_info_search_all(query_dsl: dict, page_size: int = 500, max_pages: int = 2000) -> Tuple[List[dict], int, List[str]]:
  """Fetch all results using search_after deep pagination; stable sort by _id."""
  all_hits: List[dict] = []
  urls: List[str] = []
  total: Optional[int] = None
  payload = dict(query_dsl)
  payload["size"] = page_size
  payload["sort"] = ["_id"]
  search_after = None

  for _ in range(max_pages):
    if search_after is not None:
      payload["search_after"] = search_after
    resp = requests.post(
        ASSET_INFO_URL,
        headers=_asset_info_headers(),
        auth=_asset_info_auth(),
        data=json.dumps(payload),
        timeout=60,
    )
    urls.append(getattr(resp, "url", ASSET_INFO_URL))
    resp.raise_for_status()
    data = resp.json()

    if total is None:
      th = data.get("hits", {}).get("total")
      total = th.get("value", 0) if isinstance(th, dict) else int(th or 0)

    hits = data.get("hits", {}).get("hits", []) or []
    if not hits:
      break

    all_hits.extend(hits)
    last = hits[-1]
    if "sort" in last:
      search_after = last["sort"]
    else:
      # fallback (rare)
      payload["from"] = payload.get("from", 0) + page_size
      search_after = None

    if total and len(all_hits) >= total: break
    if len(hits) < page_size: break

  if total is None:
    total = len(all_hits)
  return all_hits, total, urls

def _extract_imos(hits: List[dict]) -> List[str]:
  imos: List[str] = []
  for h in hits:
    n = _normalize_asset_hit(h)
    if n.get("imo") and n["imo"] not in imos:
      imos.append(n["imo"])
  return imos

@tool(
    "get_vessel_data",
    description=(
        "Search for vessels in Asset Information Service (OpenSearch). "
        "Accepts flexible filters like flag, name, shiptype, deadweight/year ranges, owner, builder, IMO/MMSI, or free_text. "
        "Returns IMOs (for downstream tools) and normalized records."
    ),
)
def get_vessel_data(
    flag: Optional[str] = None,
    free_text: Optional[str] = None,
    imo: Optional[str] = None,
    mmsi: Optional[str] = None,
    name: Optional[str] = None,
    call_sign: Optional[str] = None,
    shiptype: Optional[str] = None,
    dwt_min: Optional[int] = None,
    dwt_max: Optional[int] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    builder: Optional[str] = None,
    owner: Optional[str] = None,
    country_of_build: Optional[str] = None,
    extra_terms: Optional[Dict[str, str]] = None,
    page_size: Optional[int] = None,
) -> dict:
  """
  Query OpenSearch and return:
    {
      "type": "vessels",
      "imos": ["9169055", ...],
      "total": <int>,
      "records": [ {normalized vessel fields}, ... ],
      "sources": [list of request URLs per page]
    }
  """
  try:
    qdsl = _build_asset_info_query(
        free_text=free_text,
        flag=flag,
        imo=imo,
        mmsi=mmsi,
        name=name,
        call_sign=call_sign,
        shiptype=shiptype,
        dwt_min=dwt_min,
        dwt_max=dwt_max,
        year_min=year_min,
        year_max=year_max,
        builder=builder,
        owner=owner,
        country_of_build=country_of_build,
        extra_terms=extra_terms,
    )
    hits, total, urls = _asset_info_search_all(qdsl, page_size or 500)
    normalized = [_normalize_asset_hit(h) for h in hits]
    imos = []
    seen = set()
    for r in normalized:
      imo_val = r.get("imo")
      if imo_val and imo_val not in seen:
        seen.add(imo_val); imos.append(imo_val)

    return {
      "type": "vessels",
      "imos": imos,
      "total": total,
      "records": normalized,
      "sources": urls,
    }
  except Exception as e:
    return {
      "type": "vessels",
      "imos": [],
      "total": 0,
      "records": [],
      "error": f"{type(e).__name__}: {str(e)}"
    }
