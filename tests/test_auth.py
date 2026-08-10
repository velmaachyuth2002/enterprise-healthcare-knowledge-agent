from app.services.auth import hash_password, verify_password


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
