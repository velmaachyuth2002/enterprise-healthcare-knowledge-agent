from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.approval_routes import router as approval_router
from app.api.routes import router
from app.database.session import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Enterprise Healthcare Knowledge Agent", lifespan=lifespan)
app.include_router(router)
app.include_router(approval_router)

# Mounted last and at "/" so it only ever catches requests the routers
# above didn't already handle - /ask, /login, /me, /approvals, and
# FastAPI's own /docs all keep working exactly as before.
app.mount("/", StaticFiles(directory="web", html=True), name="web")
