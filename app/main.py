from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    structlog.get_logger().info("application_started", environment=settings.environment)
    yield
    structlog.get_logger().info("application_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        start = perf_counter()
        structlog.contextvars.bind_contextvars(request_id=request_id, method=request.method)
        try:
            response = await call_next(request)
        except Exception:
            structlog.get_logger().exception("request_failed", path=request.url.path)
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})
        finally:
            structlog.get_logger().info(
                "request_completed", path=request.url.path,
                duration_ms=round((perf_counter() - start) * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(router, prefix=settings.api_prefix)
    static_directory = Path(__file__).parent / "web"
    app.mount("/assets", StaticFiles(directory=static_directory), name="assets")

    @app.get("/", include_in_schema=False)
    async def web_dashboard() -> FileResponse:
        return FileResponse(static_directory / "index.html")

    return app


app = create_app()
