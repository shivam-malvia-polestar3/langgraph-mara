from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from tools_common import utcnow_iso
from graph_multi import run_turn, summarize_per_tool
from session_store import (
  SESSIONS, SESSION_CTX, SESSIONS_META,
  get_or_create_session, context_system_prompt,
  seed_defaults_from_query, update_ctx_from_tool_calls
)

app = FastAPI(title="Polestar LangGraph (Multi-Agent Router + Specialists)")

class NLQuery(BaseModel):
  query: str
  session_id: str = Field(description="Required: groups conversation under this session")
  correlation_id: str = Field(description="Required: correlation identifier for matching responses")

@app.post("/run-graph")
def run_graph(body: NLQuery):
  session_id = get_or_create_session(body.session_id)
  correlation_id = body.correlation_id

  # Seed session context from natural language (IMO/MMSI/time range/top N/duration hints)
  seed_defaults_from_query(session_id, body.query)

  # Prepare current history
  history = SESSIONS[session_id][:]

  # Mark boundary to only collect *this* turn's outputs
  before_len = len(history)

  # Execute one routed turn
  new_messages = run_turn(session_id, body.query, history)

  # Collect tool outputs for *this turn only*
  import json
  tool_results: List[dict] = []
  for m in new_messages:
    if isinstance(m, ToolMessage):
      try:
        tool_results.append(json.loads(m.content))
      except Exception:
        pass

  # Group by type and summarize
  grouped: Dict[str, List[dict]] = {}
  for r in tool_results:
    typ = r.get("type", "unknown")
    grouped.setdefault(typ, []).append(r)
  summaries = summarize_per_tool(grouped) if grouped else {}

  # Final AI text from this turn (if any)
  final_ai = next((m for m in reversed(new_messages) if getattr(m, "type", "") == "ai"), None)
  final_text = getattr(final_ai, "content", "") if final_ai else ""

  # Persist conversation (append user + new agent/tool messages)
  SESSIONS[session_id].append(HumanMessage(body.query))
  for m in new_messages:
    if isinstance(m, (AIMessage, ToolMessage)):
      SESSIONS[session_id].append(m)

  # Metadata updates
  meta = SESSIONS_META.setdefault(session_id, {"created_at": utcnow_iso(), "last_activity_at": utcnow_iso(), "first_user_message": None})
  if meta.get("first_user_message") is None:
    meta["first_user_message"] = body.query
  meta["last_activity_at"] = utcnow_iso()

  # Update session defaults (IMO/MMSI/timestamps & last_vessels)
  update_ctx_from_tool_calls(session_id, new_messages)

  # Return only this turn’s outputs
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
  lines = []
  for m in convo:
    t = getattr(m, "type", ""); c = getattr(m, "content", "")
    if isinstance(m, ToolMessage):
      import json
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

  from langchain_openai import ChatOpenAI
  from langchain_core.messages import SystemMessage, HumanMessage
  summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
  sys = SystemMessage(content="Summarize the conversation for a maritime analytics context, capturing intent, tools/data retrieved, and outcomes.")
  msg = summarizer.invoke([sys, HumanMessage(content=transcript)])
  return {"session_id": session_id, "summary": getattr(msg, "content", "")}

@app.get("/sessions/{session_id}/transcript")
def get_transcript(session_id: str):
  """Return the full conversation for UI rendering."""
  history = SESSIONS.get(session_id, [])
  out = []
  for m in history:
    role = getattr(m, "type", ""); content = getattr(m, "content", "")
    if isinstance(m, ToolMessage):
      import json
      tool_type = None
      try:
        payload = json.loads(content)
        tool_type = payload.get("type")
      except Exception:
        pass
      out.append({"role": "tool", "tool_type": tool_type, "content": content})
    elif role == "ai":
      out.append({"role": "assistant", "content": content})
    elif role == "human":
      out.append({"role": "user", "content": content})
  meta = SESSIONS_META.get(session_id, {})
  return {
    "session_id": session_id,
    "created_at": meta.get("created_at"),
    "last_activity_at": meta.get("last_activity_at"),
    "messages": out
  }

@app.get("/sessions")
def list_sessions():
  """List sessions with their first user message (for a sidebar like ChatGPT)."""
  items = []
  for sid, meta in SESSIONS_META.items():
    items.append({
      "session_id": sid,
      "first_user_message": meta.get("first_user_message"),
      "created_at": meta.get("created_at"),
      "last_activity_at": meta.get("last_activity_at"),
    })
  items.sort(key=lambda x: x.get("last_activity_at") or "", reverse=True)
  return {"sessions": items}

@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
  """Clear a session's history and context defaults."""
  SESSIONS.pop(session_id, None)
  SESSION_CTX.pop(session_id, None)
  SESSIONS_META.pop(session_id, None)
  return {"session_id": session_id, "cleared": True}

@app.get("/healthz")
def healthz():
  return {"ok": True}
