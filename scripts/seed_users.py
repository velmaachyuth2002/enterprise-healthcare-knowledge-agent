"""Creates the first manager and employee accounts for manual testing.

There's no sign-up endpoint by design - internal employees are provisioned
by an admin in a real deployment, not self-registered. Run with:

    uv run python -m scripts.seed_users

Safe to re-run: skips any account whose email already exists.
"""

from app.database.session import Base, SessionLocal, engine
from app.models.user import User, UserRole
from app.services.auth import hash_password

_SEED_USERS = [
    {
        "email": "manager@medflow.example",
        "name": "Morgan Reyes",
        "password": "manager-dev-pass",
        "role": UserRole.MANAGER,
    },
    {
        "email": "employee@medflow.example",
        "name": "Alex Chen",
        "password": "employee-dev-pass",
        "role": UserRole.EMPLOYEE,
    },
]


def seed() -> None:
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        for spec in _SEED_USERS:
            if session.query(User).filter_by(email=spec["email"]).first() is not None:
                print(f"skip (already exists): {spec['email']}")
                continue
            session.add(
                User(
                    email=spec["email"],
                    name=spec["name"],
                    hashed_password=hash_password(spec["password"]),
                    role=spec["role"],
                )
            )
            session.commit()
            print(f"created {spec['role'].value}: {spec['email']} / {spec['password']}")
    finally:
        session.close()


if __name__ == "__main__":
    seed()
