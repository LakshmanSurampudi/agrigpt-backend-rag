from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from services.clip_ingest_service import clip_ingest_service

router = APIRouter(tags=["CLIP Ingestion"])


class IngestResponse(BaseModel):
    message: str
    filename: str
    text_chunks: int
    images_processed: int
    images_stored: int
    errors: List[str]


class ImageQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class ImageResult(BaseModel):
    score: float
    image_url: str
    source: str
    page: int
    image_index: int


class ImageQueryResponse(BaseModel):
    query: str
    results: List[ImageResult]
    count: int


@router.post("/clip-ingest-data", response_model=IngestResponse)
async def ingest_pdf_with_clip(file: UploadFile = File(...)):
    """
    Ingest PDF with CLIP processing:
    1. Extract text and images separately
    2. Embed text with Google text-embedding-004
    3. Embed images with CLIP
    4. Store images in Cloudflare R2
    5. Store all embeddings in Pinecone with image URLs as metadata
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the PDF
        results = await clip_ingest_service.process_pdf(temp_file_path, file.filename)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return IngestResponse(
            message=f"PDF processed successfully. {results['text_chunks']} text chunks, {results['images_stored']} images stored.",
            filename=results["filename"],
            text_chunks=results["text_chunks"],
            images_processed=results["images_processed"],
            images_stored=results["images_stored"],
            errors=results["errors"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/clip-query-data", response_model=ImageQueryResponse)
async def query_clip_images(request: ImageQueryRequest):
    """
    Query CLIP index for relevant images based on text:
    1. Converts text query to CLIP embedding
    2. Searches Pinecone CLIP index for similar image embeddings
    3. Returns matching images with R2 URLs
    """
    try:
        images = await clip_ingest_service.query_images(request.query, request.top_k)
        
        return ImageQueryResponse(
            query=request.query,
            results=[ImageResult(**img) for img in images],
            count=len(images)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying images: {str(e)}")


class AskWithImageResponse(BaseModel):
    answer: str
    matched_sources: List[str]
    related_images: List[str]
    confidence: float


@router.post("/ask-with-image", response_model=AskWithImageResponse)
async def ask_with_image(
    file: UploadFile = File(...),
    query: str = "What disease does this crop have and how can I treat it?"
):
    """
    Answer questions about a crop image:
    1. Upload a crop image
    2. CLIP finds similar disease images in the database
    3. Retrieves related text context
    4. LLM generates an expert answer
    """
    # Validate file type by extension
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, WebP) are allowed")
    
    try:
        image_bytes = await file.read()
        
        result = await clip_ingest_service.ask_with_image(image_bytes, query)
        
        return AskWithImageResponse(
            answer=result["answer"],
            matched_sources=result["matched_sources"],
            related_images=result["related_images"],
            confidence=result["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image query: {str(e)}")
