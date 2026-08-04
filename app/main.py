from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Enterprise Healthcare Knowledge Agent")
app.include_router(router)
