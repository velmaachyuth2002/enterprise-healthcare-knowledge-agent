from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.session import Base


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    # StaticPool + check_same_thread=False: an in-memory SQLite database is
    # private to the connection that opened it, and the default pool doesn't
    # guarantee the same connection is reused across threads. That's
    # invisible when a test calls the graph directly (one thread throughout),
    # but FastAPI's TestClient runs sync route handlers in a worker thread -
    # without this, that thread can get handed a different, empty database
    # than the one this fixture just created tables in.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
