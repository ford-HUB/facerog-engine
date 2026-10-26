import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.dependencies import ServiceContainer, build_container
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging

WEB_DIR = Path(__file__).parent / "web"

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    container_factory: Callable[[Settings], ServiceContainer] = build_container,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = container_factory(settings)
        container.enrollment_service.enroll_dataset()
        if settings.video_autostart:
            container.tracking_service.start()
        app.state.container = container
        yield
        container.tracking_service.stop()

    app = FastAPI(title="Face Recognition Engine", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(api_router)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/tv", include_in_schema=False)
    def tv_mode() -> FileResponse:
        return FileResponse(WEB_DIR / "tv.html")

    return app


configure_logging()
app = create_app()
