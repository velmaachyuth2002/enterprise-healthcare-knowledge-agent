from datetime import datetime, timedelta

import bcrypt
import jwt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


class InvalidTokenError(Exception):
    """Raised for any token that fails to decode: expired, malformed, or
    signed with a different secret. Callers don't need to distinguish which
    - all three mean "this request isn't authenticated"."""


def create_access_token(
    user_id: int, *, secret: str, algorithm: str, expires_minutes: int, now: datetime
) -> str:
    # `now` is injected (not datetime.now() called directly) for the same
    # reason `today()` is injected in agent_graph.py: testability, here
    # specifically so an already-expired token can be constructed
    # deterministically rather than waiting on the clock.
    #
    # `now` must be timezone-aware UTC (datetime.now(UTC), not
    # datetime.now()) - PyJWT computes `exp` via utctimetuple(), which
    # treats a naive datetime as if it were already UTC rather than
    # converting it. Passing local naive time silently shifts every
    # token's real expiry by the system's UTC offset.
    if not secret:
        # Fails clearly here instead of as a raw InvalidKeyError from
        # PyJWT three frames down - JWT_SECRET being unset in .env is a
        # setup mistake, not something that should look like a library bug.
        raise ValueError("JWT secret must not be empty - set JWT_SECRET in .env")
    payload = {"sub": str(user_id), "exp": now + timedelta(minutes=expires_minutes)}
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(token: str, *, secret: str, algorithm: str) -> int:
    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError from exc
    return int(payload["sub"])
