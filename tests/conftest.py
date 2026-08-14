from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.session import Base

# Imported for their side effect - registering each table on Base.metadata
# - not for the names themselves. A SQLAlchemy model only gets registered
# when its module is actually imported somewhere; db_session's create_all()
# below has always relied on *some* test file happening to import every
# model first. That's fragile: it only worked because pytest collects the
# whole tests/ directory by default, and broke the moment a new test file
# was run in isolation without importing the models it needed transitively.
from app.models import approval_request, llm_usage, ticket, user  # noqa: F401
from app.services.document_index import DocumentIndex

_DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"


@pytest.fixture(scope="session")
def document_index() -> DocumentIndex:
    # Session-scoped: loading the sentence-transformers model is the slow
    # part (roughly a second), and it's the same model/corpus for every
    # test that needs it - rebuilding per-test would make the suite slow
    # for no correctness benefit.
    return DocumentIndex(_DOCUMENTS_DIR)


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
