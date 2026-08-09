from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends
from groq import Groq
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.graph.agent_graph import build_graph
from app.services.document_index import DocumentIndex
from app.services.llm_gateway import LlmGateway
from app.tools.document_search_tool import DocumentSearchTool
from app.tools.sql_tool import TicketCountTool, UnresolvedTicketsTool

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str


@lru_cache
def get_groq_client() -> Groq:
    # Safe to cache, unlike get_agent_graph below: the client itself is
    # stateless (no per-request resource attached to it), so reusing the
    # same instance across requests is correct, not just convenient.
    return Groq(api_key=get_settings().groq_api_key)


@lru_cache
def get_document_search_tool() -> DocumentSearchTool:
    # Cached like get_groq_client above, for the same reason: unlike the SQL
    # tools, this one holds no per-request Session - it's built from the
    # documents directory, which doesn't change between requests. Rebuilding
    # it (reloading the embedding model, re-embedding every document) on
    # every single call would be pure waste.
    settings = get_settings()
    return DocumentSearchTool(DocumentIndex(Path(settings.documents_dir)))


def get_agent_graph(
    session: Session = Depends(get_db),
    client: Groq = Depends(get_groq_client),
    document_search_tool: DocumentSearchTool = Depends(get_document_search_tool),
):
    # Not cached: two of these three tools, and the gateway itself, hold a
    # per-request Session that get_db() closes at the end of this request,
    # so a graph built from it can't be reused by a later request.
    settings = get_settings()
    gateway = LlmGateway(client, settings.groq_model, session, feature="planner_tool_selection")
    return build_graph(
        document_search_tool,
        TicketCountTool(session),
        UnresolvedTicketsTool(session),
        gateway,
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, graph=Depends(get_agent_graph)) -> AskResponse:
    result = graph.invoke({"question": request.question})
    return AskResponse(answer=result["answer"])
