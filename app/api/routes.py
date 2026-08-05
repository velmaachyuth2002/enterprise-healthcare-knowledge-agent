from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.graph.agent_graph import build_graph
from app.tools.policy_tool import PolicyTool
from app.tools.sql_tool import TicketCountTool, UnresolvedTicketsTool

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str


def get_agent_graph(session: Session = Depends(get_db)):
    # Not cached: two of these three tools hold a per-request Session that
    # get_db() closes at the end of this request, so a graph built from it
    # can't be reused by a later request.
    return build_graph(
        PolicyTool(),
        TicketCountTool(session),
        UnresolvedTicketsTool(session),
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, graph=Depends(get_agent_graph)) -> AskResponse:
    result = graph.invoke({"question": request.question})
    return AskResponse(answer=result["answer"])
