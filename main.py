from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import tempfile
from typing import List, Optional

# Load environment variables
load_dotenv()

# Import RAG components
from rag_service import RAGService

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG service on startup"""
    await rag_service.initialize()

@app.post("/upload-crop-data", response_model=dict)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the PDF
        num_chunks = await rag_service.process_pdf(temp_file_path, file.filename)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return {
            "message": f"PDF processed successfully. Added {num_chunks} chunks to knowledge base.",
            "filename": file.filename,
            "chunks": num_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-consultant", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system"""
    try:
        response, sources = await rag_service.query(request.query, request.chat_history)
        return ChatResponse(response=response, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/upload-government-schemes", response_model=dict)
async def upload_government_schemes(file: UploadFile = File(...)):
    """Upload and process government schemes file"""
    return {}

@app.post("/query-government-schemes", response_model=ChatResponse)
async def query_government_schemes(request: ChatRequest):
    """Query government schemes"""
    return ChatResponse(response="", sources=[])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Chatbot API"}

@app.delete("/clear-knowledge-base")
async def clear_knowledge_base():
    """Clear all documents from the knowledge base"""
    try:
        await rag_service.clear_knowledge_base()
        return {"message": "Knowledge base cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)