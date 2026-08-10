from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_role
from app.core.config import Settings
from app.models.user import User, UserRole
from app.services.auth import create_access_token

_SETTINGS = Settings(
    jwt_secret="test-secret-that-is-long-enough-to-avoid-hmac-warnings",
    jwt_algorithm="HS256",
    jwt_expiry_minutes=10,
)


def _token_for(user_id: int) -> str:
    return create_access_token(
        user_id,
        secret=_SETTINGS.jwt_secret,
        algorithm=_SETTINGS.jwt_algorithm,
        expires_minutes=_SETTINGS.jwt_expiry_minutes,
        now=datetime.now(UTC),
    )


def _add_user(db_session: Session, role: UserRole) -> User:
    user = User(
        email="user@medflow.example", name="Test User", hashed_password="irrelevant", role=role
    )
    db_session.add(user)
    db_session.commit()
    return user


def test_get_current_user_returns_the_user_for_a_valid_token(db_session: Session) -> None:
    user = _add_user(db_session, UserRole.EMPLOYEE)

    result = get_current_user(token=_token_for(user.id), session=db_session, settings=_SETTINGS)

    assert result.id == user.id


def test_get_current_user_rejects_an_invalid_token(db_session: Session) -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token="not-a-real-token", session=db_session, settings=_SETTINGS)

    assert exc_info.value.status_code == 401


def test_get_current_user_rejects_a_token_for_a_deleted_user(db_session: Session) -> None:
    # A valid, unexpired token whose user_id no longer has a matching row -
    # the account could have been removed after the token was issued.
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token=_token_for(999), session=db_session, settings=_SETTINGS)

    assert exc_info.value.status_code == 401


def test_require_role_allows_a_matching_role(db_session: Session) -> None:
    manager = _add_user(db_session, UserRole.MANAGER)
    check = require_role(UserRole.MANAGER)

    result = check(current_user=manager)

    assert result is manager


def test_require_role_rejects_a_mismatched_role(db_session: Session) -> None:
    employee = _add_user(db_session, UserRole.EMPLOYEE)
    check = require_role(UserRole.MANAGER)

    with pytest.raises(HTTPException) as exc_info:
        check(current_user=employee)

    assert exc_info.value.status_code == 403
