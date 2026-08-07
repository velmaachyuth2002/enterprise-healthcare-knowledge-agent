from functools import lru_cache

from fastapi import APIRouter, Depends
from groq import Groq
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.graph.agent_graph import build_graph
from app.services.llm_gateway import LlmGateway
from app.tools.policy_tool import PolicyTool
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


def get_agent_graph(session: Session = Depends(get_db), client: Groq = Depends(get_groq_client)):
    # Not cached: two of these three tools, and the gateway itself, hold a
    # per-request Session that get_db() closes at the end of this request,
    # so a graph built from it can't be reused by a later request.
    settings = get_settings()
    gateway = LlmGateway(client, settings.groq_model, session, feature="planner_tool_selection")
    return build_graph(
        PolicyTool(),
        TicketCountTool(session),
        UnresolvedTicketsTool(session),
        gateway,
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, graph=Depends(get_agent_graph)) -> AskResponse:
    result = graph.invoke({"question": request.question})
    return AskResponse(answer=result["answer"])
