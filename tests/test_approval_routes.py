from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_email_service
from app.database.session import get_db
from app.main import app
from app.models.approval_request import ApprovalRequest, ApprovalStatus
from app.models.ticket import Ticket, TicketPriority, TicketStatus
from app.models.user import User, UserRole


class _FakeEmailService:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append((to, subject, body))


def _override_get_db(session: Session):
    def _get_db():
        yield session

    return _get_db


def _authenticate_as(user: User):
    return lambda: user


def _seed(db_session: Session) -> tuple[Ticket, User, User, ApprovalRequest]:
    ticket = Ticket(
        subject="Claims submission blocked",
        status=TicketStatus.OPEN,
        priority=TicketPriority.MEDIUM,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    employee = User(
        email="employee@medflow.example",
        name="Alex Chen",
        hashed_password="x",
        role=UserRole.EMPLOYEE,
    )
    manager = User(
        email="manager@medflow.example",
        name="Morgan Reyes",
        hashed_password="x",
        role=UserRole.MANAGER,
    )
    db_session.add_all([ticket, employee, manager])
    db_session.commit()

    approval = ApprovalRequest(
        ticket_id=ticket.id,
        requester_id=employee.id,
        requested_priority=TicketPriority.URGENT,
        reason="Affecting claims submission",
        requested_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    db_session.add(approval)
    db_session.commit()

    return ticket, employee, manager, approval


def _client(db_session: Session, current_user: User, email_service=None) -> TestClient:
    app.dependency_overrides[get_db] = _override_get_db(db_session)
    app.dependency_overrides[get_current_user] = _authenticate_as(current_user)
    app.dependency_overrides[get_email_service] = lambda: email_service or _FakeEmailService()
    return TestClient(app)


def test_list_pending_approvals_returns_only_pending_requests(db_session: Session) -> None:
    ticket, employee, manager, approval = _seed(db_session)
    decided = ApprovalRequest(
        ticket_id=ticket.id,
        requester_id=employee.id,
        requested_priority=TicketPriority.HIGH,
        reason="Already handled",
        status=ApprovalStatus.APPROVED,
        requested_at=datetime(2026, 7, 1, tzinfo=UTC),
        decided_by_id=manager.id,
        decided_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    db_session.add(decided)
    db_session.commit()

    client = _client(db_session, manager)
    try:
        response = client.get("/approvals")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == approval.id


def test_list_pending_approvals_requires_manager_role(db_session: Session) -> None:
    _, employee, _manager, _approval = _seed(db_session)

    client = _client(db_session, employee)
    try:
        response = client.get("/approvals")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403


def test_list_pending_approvals_requires_authentication() -> None:
    client = TestClient(app)

    response = client.get("/approvals")

    assert response.status_code == 401


def test_decide_approve_updates_ticket_priority_and_notifies_requester(
    db_session: Session,
) -> None:
    ticket, employee, manager, approval = _seed(db_session)
    fake_email = _FakeEmailService()

    client = _client(db_session, manager, fake_email)
    try:
        response = client.post(f"/approvals/{approval.id}/decide", json={"approved": True})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "approved"

    db_session.refresh(ticket)
    db_session.refresh(approval)
    assert ticket.priority == TicketPriority.URGENT
    assert approval.status == ApprovalStatus.APPROVED
    assert approval.decided_by_id == manager.id
    assert approval.decided_at is not None

    assert len(fake_email.sent) == 1
    to, subject, body = fake_email.sent[0]
    assert to == employee.email
    assert "approved" in subject
    assert str(ticket.id) in body


def test_decide_reject_does_not_change_ticket_priority(db_session: Session) -> None:
    ticket, employee, manager, approval = _seed(db_session)
    fake_email = _FakeEmailService()

    client = _client(db_session, manager, fake_email)
    try:
        response = client.post(f"/approvals/{approval.id}/decide", json={"approved": False})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    db_session.refresh(ticket)
    db_session.refresh(approval)
    assert ticket.priority == TicketPriority.MEDIUM
    assert approval.status == ApprovalStatus.REJECTED

    assert len(fake_email.sent) == 1
    assert fake_email.sent[0][0] == employee.email


def test_decide_requires_manager_role(db_session: Session) -> None:
    ticket, employee, _manager, approval = _seed(db_session)

    client = _client(db_session, employee)
    try:
        response = client.post(f"/approvals/{approval.id}/decide", json={"approved": True})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    db_session.refresh(ticket)
    assert ticket.priority == TicketPriority.MEDIUM


def test_decide_requires_authentication(db_session: Session) -> None:
    _, _employee, _manager, approval = _seed(db_session)
    app.dependency_overrides[get_db] = _override_get_db(db_session)
    client = TestClient(app)

    try:
        response = client.post(f"/approvals/{approval.id}/decide", json={"approved": True})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_decide_returns_404_for_a_nonexistent_approval(db_session: Session) -> None:
    _, _employee, manager, _approval = _seed(db_session)

    client = _client(db_session, manager)
    try:
        response = client.post("/approvals/999/decide", json={"approved": True})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_decide_returns_409_when_already_decided(db_session: Session) -> None:
    ticket, _employee, manager, approval = _seed(db_session)

    client = _client(db_session, manager)
    try:
        first = client.post(f"/approvals/{approval.id}/decide", json={"approved": True})
        second = client.post(f"/approvals/{approval.id}/decide", json={"approved": False})
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == 200
    assert second.status_code == 409

    # The second (rejected) call must not have overwritten the first
    # decision's effect on the ticket.
    db_session.refresh(ticket)
    assert ticket.priority == TicketPriority.URGENT
