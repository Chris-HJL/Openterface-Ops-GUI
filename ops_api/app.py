"""
FastAPI application configuration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .endpoints import router

def create_app() -> FastAPI:
    """
    Create FastAPI application

    Returns:
        FastAPI application instance
    """
    # Create FastAPI application
    app = FastAPI(title="Openterface Ops API", version="2.0.0")

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=".", html=True), name="static")

    # Register routes
    app.include_router(router)

    return app
