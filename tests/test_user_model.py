import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.user import User, UserRole


def test_user_round_trips_through_the_database(db_session: Session) -> None:
    db_session.add(
        User(
            email="manager@medflow.example",
            name="Morgan Reyes",
            hashed_password="not-a-real-hash",
            role=UserRole.MANAGER,
        )
    )
    db_session.commit()

    user = db_session.query(User).one()

    assert user.email == "manager@medflow.example"
    assert user.name == "Morgan Reyes"
    assert user.role == UserRole.MANAGER


def test_role_is_stored_as_its_lowercase_value_not_the_enum_name(db_session: Session) -> None:
    # Same regression this guards against as TicketStatus/TicketPriority:
    # SQLAlchemy's default Enum behavior persists the member *name*
    # ("MANAGER") rather than its value ("manager") unless told otherwise.
    # A future role check comparing against "manager" would silently fail
    # if this regressed.
    db_session.add(
        User(
            email="manager@medflow.example",
            name="Morgan Reyes",
            hashed_password="not-a-real-hash",
            role=UserRole.MANAGER,
        )
    )
    db_session.commit()

    raw_role = db_session.execute(text("SELECT role FROM users")).scalar_one()

    assert raw_role == "manager"


def test_email_must_be_unique(db_session: Session) -> None:
    db_session.add(
        User(
            email="duplicate@medflow.example",
            name="First User",
            hashed_password="not-a-real-hash",
            role=UserRole.EMPLOYEE,
        )
    )
    db_session.commit()

    db_session.add(
        User(
            email="duplicate@medflow.example",
            name="Second User",
            hashed_password="not-a-real-hash",
            role=UserRole.EMPLOYEE,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
