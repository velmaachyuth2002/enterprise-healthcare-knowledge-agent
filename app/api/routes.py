from functools import lru_cache

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.graph.agent_graph import build_graph
from app.tools.policy_tool import PolicyTool

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str


@lru_cache
def get_agent_graph():
    return build_graph(PolicyTool())


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, graph=Depends(get_agent_graph)) -> AskResponse:
    result = graph.invoke({"question": request.question})
    return AskResponse(answer=result["answer"])
