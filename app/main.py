from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scalar_fastapi import get_scalar_api_reference
from app.routers import estimations

app = FastAPI(
    title="CAG Estimator",
    description="API for software project estimation using CAG architecture",
    version="0.1.0",
    docs_url=None # In order to deactivate /docs swagger default
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(estimations.router)

@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title
    )

@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": "cag-estimator"}
    )