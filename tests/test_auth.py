from datetime import UTC, datetime

import pytest

from app.services.auth import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

_SECRET = "test-secret-that-is-long-enough-to-avoid-hmac-warnings"
_ALGORITHM = "HS256"


def test_hash_password_does_not_store_the_plaintext() -> None:
    hashed = hash_password("correct-horse-battery-staple")

    assert hashed != "correct-horse-battery-staple"


def test_verify_password_accepts_the_correct_password() -> None:
    hashed = hash_password("correct-horse-battery-staple")

    assert verify_password("correct-horse-battery-staple", hashed) is True


def test_verify_password_rejects_the_wrong_password() -> None:
    hashed = hash_password("correct-horse-battery-staple")

    assert verify_password("wrong-password", hashed) is False


def test_hashing_the_same_password_twice_produces_different_hashes() -> None:
    # bcrypt salts each hash randomly - if this regressed to an unsalted
    # scheme, two users with the same password would have identical hashes,
    # letting an attacker with DB read access spot password reuse instantly.
    first = hash_password("correct-horse-battery-staple")
    second = hash_password("correct-horse-battery-staple")

    assert first != second


def test_decode_returns_the_user_id_encoded_at_creation() -> None:
    token = create_access_token(
        42, secret=_SECRET, algorithm=_ALGORITHM, expires_minutes=10, now=datetime.now(UTC)
    )

    user_id = decode_access_token(token, secret=_SECRET, algorithm=_ALGORITHM)

    assert user_id == 42


def test_decode_rejects_a_token_signed_with_a_different_secret() -> None:
    token = create_access_token(
        42, secret=_SECRET, algorithm=_ALGORITHM, expires_minutes=10, now=datetime.now(UTC)
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(token, secret="a-different-secret", algorithm=_ALGORITHM)


def test_decode_rejects_an_expired_token() -> None:
    # `now` is set far enough in the past that expires_minutes later is
    # still before the real current time - no need to fake the clock at
    # decode time, real wall-clock is always "later" than this.
    token = create_access_token(
        42,
        secret=_SECRET,
        algorithm=_ALGORITHM,
        expires_minutes=10,
        now=datetime(2020, 1, 1),
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(token, secret=_SECRET, algorithm=_ALGORITHM)


def test_decode_rejects_a_malformed_token() -> None:
    with pytest.raises(InvalidTokenError):
        decode_access_token("not-a-real-token", secret=_SECRET, algorithm=_ALGORITHM)


def test_create_access_token_rejects_an_empty_secret() -> None:
    # Guards against the confusing failure mode this actually hit during
    # development: an unset JWT_SECRET silently becoming "" and only
    # surfacing as a raw PyJWT InvalidKeyError deep in the call stack.
    with pytest.raises(ValueError, match="JWT secret"):
        create_access_token(
            42, secret="", algorithm=_ALGORITHM, expires_minutes=10, now=datetime.now(UTC)
        )
