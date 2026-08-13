from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_email_service
from app.core.config import get_settings
from app.database.session import get_db
from app.graph.agent_graph import build_graph
from app.models.user import User, UserRole
from app.services.auth import create_access_token, verify_password
from app.services.document_index import DocumentIndex
from app.services.email_service import EmailService
from app.services.llm_gateway import LlmGateway
from app.tools.document_search_tool import DocumentSearchTool
from app.tools.escalate_ticket_tool import EscalateTicketTool
from app.tools.sql_tool import TicketCountTool, UnresolvedTicketsTool

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    role: UserRole


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    # The only thing a token proves is *which user* - it doesn't carry role
    # or name, so the frontend needs a real call to know who's logged in
    # and which view (employee chat vs. manager approvals) to show.
    return UserResponse.model_validate(current_user)


@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_db)
) -> TokenResponse:
    # form_data.username holds the email - that's the OAuth2 password-flow
    # field name FastAPI's docs UI expects, not a statement that email and
    # username are different things here.
    user = session.query(User).filter_by(email=form_data.username).first()
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password"
        )

    settings = get_settings()
    token = create_access_token(
        user.id,
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        expires_minutes=settings.jwt_expiry_minutes,
        now=datetime.now(UTC),
    )
    return TokenResponse(access_token=token)


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
    email_service: EmailService = Depends(get_email_service),
):
    # Not cached: most of these tools, and the gateway itself, hold a
    # per-request Session that get_db() closes at the end of this request,
    # so a graph built from it can't be reused by a later request.
    settings = get_settings()
    gateway = LlmGateway(client, settings.groq_model, session, feature="planner_tool_selection")

    def get_manager_emails() -> list[str]:
        # Deferred rather than queried eagerly here: this only needs to run
        # on the (rare) escalation path, not on every /ask request.
        return [user.email for user in session.query(User).filter_by(role=UserRole.MANAGER).all()]

    return build_graph(
        document_search_tool,
        TicketCountTool(session),
        UnresolvedTicketsTool(session),
        EscalateTicketTool(session),
        email_service,
        get_manager_emails,
        gateway,
    )


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    current_user: User = Depends(get_current_user),
    graph=Depends(get_agent_graph),
) -> AskResponse:
    # requester_id comes only from the verified token, never from the
    # request body or anything the LLM parses - see EscalateTicketTool for
    # why that boundary matters.
    result = graph.invoke({"question": request.question, "requester_id": current_user.id})
    return AskResponse(answer=result["answer"])
