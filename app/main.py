from fastapi import FastAPI

from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}
