import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import ENV_PROJECT
from app.modules.gemini_processor import (
    GeminiWorker,
    RateLimitManager,
    clear_gemini_queues,
)
from app.routes.router import routers
from app.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting background services...")

    rate_limit_manager = RateLimitManager()
    asyncio.create_task(rate_limit_manager.run())

    num_workers = 3
    for i in range(num_workers):
        worker = GeminiWorker()
        asyncio.create_task(worker.run())

    logger.info(f"Started Rate Limit Manager and {num_workers} Gemini workers")

    yield  # app runs here

    logger.info("Shutting down background services...")
    await clear_gemini_queues()
    logger.info("Background services shutdown complete")


app = FastAPI(title=ENV_PROJECT.PROJECT_NAME, lifespan=lifespan)
app.include_router(routers)

allowed_origins = [
    "http://localhost:3000",
    "https://navacord.netlify.app",
    "https://documentxintelligence.netlify.app",
    "https://docx.9ai.in",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(
    f"FastAPI application '{ENV_PROJECT.PROJECT_NAME}' initialized with CORS middleware"
)


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health_check():
    logger.debug("Health check endpoint accessed")
    return {"message": "Agent-Server is running!"}
