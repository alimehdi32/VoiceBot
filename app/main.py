from fastapi import FastAPI
from app.api.routes.voicebot import router
from app.core.config import settings

app = FastAPI(

    title=settings.PROJECT_NAME,

    version=settings.API_VERSION
)

app.include_router(router)