from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Routers and Services
from routes.rag_routes import router as rag_router, rag_service
from routes.clip_ingest_routes import router as clip_ingest_router
from services.clip_ingest_service import clip_ingest_service

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Include Routers
app.include_router(rag_router)
app.include_router(clip_ingest_router)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await rag_service.initialize()
    await clip_ingest_service.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
