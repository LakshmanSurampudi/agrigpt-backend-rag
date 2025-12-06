import os
import io
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

# PDF and Image processing
import fitz  # pymupdf
from PIL import Image

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# CLIP Embeddings
from sentence_transformers import SentenceTransformer

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangSmith
from langsmith import traceable

# R2 Storage
from services.r2_storage_service import r2_storage

load_dotenv()


class ClipIngestService:
    """
    CLIP-based PDF ingestion service that:
    1. Extracts text and images separately from PDFs
    2. Embeds text with Google text-embedding-004
    3. Embeds images with CLIP
    4. Stores images in Cloudflare R2
    5. Stores embeddings in Pinecone with image URLs as metadata
    """
    
    def __init__(self):
        self.text_embeddings = None
        self.clip_model = None
        self.text_vectorstore = None
        self.pinecone_client = None
        self.clip_index = None
        self.llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        print("Initializing CLIP Ingest Service...")
        
        try:
            # Initialize R2 storage (optional - continues if fails)
            try:
                r2_storage.initialize()
            except Exception as e:
                print(f"Warning: R2 storage not configured: {e}")
                print("Image storage to R2 will be disabled. Images will still be embedded but not stored.")
            
            # Initialize text embeddings (Google)
            self.text_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            print("Text embeddings initialized (Google text-embedding-004)")
            
            # Initialize CLIP model for image embeddings
            print("Loading CLIP model (this may take a moment)...")
            self.clip_model = SentenceTransformer("clip-ViT-B-32")
            print("CLIP model initialized (clip-ViT-B-32, 512 dimensions)")
            
            # Initialize LLM for generating answers
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3
            )
            print("LLM initialized (gemini-2.0-flash)")
            
            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            # Setup text index (768 dimensions for Google embeddings)
            text_index_name = os.getenv("PINECONE_INDEX", "agrigpt-backend-rag-index")
            self._ensure_index_exists(text_index_name, 768)
            
            self.text_vectorstore = PineconeVectorStore(
                index_name=text_index_name,
                embedding=self.text_embeddings,
                pinecone_api_key=os.getenv("PINECONE_API_KEY")
            )
            print(f"Text vector store initialized: {text_index_name}")
            
            # Setup CLIP index (512 dimensions for CLIP embeddings)
            clip_index_name = os.getenv("PINECONE_CLIP_INDEX", "agrigpt-backend-rag-clip-index")
            self._ensure_index_exists(clip_index_name, 512)
            self.clip_index = self.pinecone_client.Index(clip_index_name)
            print(f"CLIP index initialized: {clip_index_name}")
            
            self.initialized = True
            print("CLIP Ingest Service initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing CLIP Ingest Service: {str(e)}")
            raise e
    
    def _ensure_index_exists(self, index_name: str, dimension: int):
        """Ensure Pinecone index exists, create if not"""
        existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index: {index_name} (dimension={dimension})")
            self.pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract all text from PDF"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    
    def extract_images_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF with page information
        
        Returns:
            List of dicts with 'image_bytes', 'page_num', 'image_index'
        """
        images = []
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PNG if needed for consistency
                    if image_ext.lower() not in ["png", "jpg", "jpeg"]:
                        img = Image.open(io.BytesIO(image_bytes))
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        image_ext = "png"
                    
                    images.append({
                        "image_bytes": image_bytes,
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "ext": image_ext
                    })
                    
                except Exception as e:
                    print(f"Error extracting image from page {page_num + 1}: {str(e)}")
                    continue
        
        doc.close()
        print(f"Extracted {len(images)} images from PDF")
        return images
    
    def embed_image(self, image_bytes: bytes) -> List[float]:
        """Embed image using CLIP model"""
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # CLIP expects RGB images
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Get embedding
        embedding = self.clip_model.encode(img)
        return embedding.tolist()
    
    @traceable(run_type="chain")
    async def process_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process PDF: extract text and images, embed and store all
        
        Returns:
            Dict with processing results
        """
        if not self.initialized:
            raise Exception("CLIP Ingest Service not initialized")
        
        results = {
            "filename": filename,
            "text_chunks": 0,
            "images_processed": 0,
            "images_stored": 0,
            "errors": []
        }
        
        try:
            # 1. Extract and process text
            print(f"Processing PDF: {filename}")
            text = self.extract_text_from_pdf(file_path)
            
            if text.strip():
                chunks = self.text_splitter.split_text(text)
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk": i,
                            "total_chunks": len(chunks),
                            "type": "text"
                        }
                    )
                    for i, chunk in enumerate(chunks)
                ]
                
                self.text_vectorstore.add_documents(documents)
                results["text_chunks"] = len(chunks)
                print(f"Stored {len(chunks)} text chunks")
            
            # 2. Extract and process images
            images = self.extract_images_from_pdf(file_path)
            results["images_processed"] = len(images)
            
            for img_data in images:
                try:
                    # Try to upload to R2 (optional)
                    img_filename = f"{filename}_p{img_data['page_num']}_i{img_data['image_index']}.{img_data['ext']}"
                    r2_url = r2_storage.upload_image(
                        img_data["image_bytes"],
                        img_filename,
                        content_type=f"image/{img_data['ext']}"
                    )
                    
                    # Generate CLIP embedding (always)
                    clip_embedding = self.embed_image(img_data["image_bytes"])
                    
                    # Store in Pinecone with metadata
                    vector_id = f"{filename}_img_{img_data['page_num']}_{img_data['image_index']}"
                    
                    self.clip_index.upsert(
                        vectors=[{
                            "id": vector_id,
                            "values": clip_embedding,
                            "metadata": {
                                "source": filename,
                                "page": img_data["page_num"],
                                "image_index": img_data["image_index"],
                                "image_url": r2_url or "",  # Empty if R2 not configured
                                "type": "image"
                            }
                        }]
                    )
                    
                    results["images_stored"] += 1
                    if r2_url:
                        print(f"Stored image with R2: {vector_id} -> {r2_url}")
                    else:
                        print(f"Stored image embedding (no R2): {vector_id}")
                    
                except Exception as e:
                    error_msg = f"Error processing image {img_data['page_num']}-{img_data['image_index']}: {str(e)}"
                    print(error_msg)
                    results["errors"].append(error_msg)
            
            print(f"PDF processing complete: {results}")
            return results
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    @traceable(run_type="chain")
    async def query_images(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query CLIP index for relevant images based on text
        
        Args:
            query: Text query to search for
            top_k: Number of results to return
            
        Returns:
            List of matching images with URLs and metadata
        """
        if not self.initialized:
            raise Exception("CLIP Ingest Service not initialized")
        
        try:
            # Embed the query text using CLIP
            query_embedding = self.clip_model.encode(query).tolist()
            
            # Search Pinecone CLIP index
            results = self.clip_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            images = []
            for match in results.matches:
                images.append({
                    "score": match.score,
                    "image_url": match.metadata.get("image_url", ""),
                    "source": match.metadata.get("source", ""),
                    "page": match.metadata.get("page", 0),
                    "image_index": match.metadata.get("image_index", 0)
                })
            
            return images
            
        except Exception as e:
            raise Exception(f"Error querying CLIP index: {str(e)}")
    
    @traceable(run_type="chain")
    async def ask_with_image(self, image_bytes: bytes, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a question about a crop image using CLIP matching + text context
        
        Args:
            image_bytes: The uploaded image as bytes
            query: User's question about the image
            top_k: Number of similar images to retrieve
            
        Returns:
            Dict with answer, matched sources, and image URLs
        """
        if not self.initialized:
            raise Exception("CLIP Ingest Service not initialized")
        
        try:
            # 1. Load and embed the uploaded image with CLIP
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            image_embedding = self.clip_model.encode(img).tolist()
            
            # 2. Find similar images in CLIP index
            matched_sources = []
            image_urls = []
            confidence = 0
            try:
                clip_results = self.clip_index.query(
                    vector=image_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                
                for match in clip_results.matches:
                    source = match.metadata.get("source", "")
                    matched_sources.append(source)
                    image_url = match.metadata.get("image_url", "")
                    if image_url:
                        image_urls.append(image_url)
                        
                confidence = clip_results.matches[0].score if clip_results.matches else 0
            except:
                pass
            
            # 3. Retrieve related text context from text index
            docs = self.text_vectorstore.similarity_search(query, k=5)
            text_context = "\n\n".join([doc.page_content for doc in docs])
            
            # 4. Generate answer using LLM with text context only
            prompt = f"""You are an agricultural expert helping a farmer identify crop diseases.

The farmer has uploaded an image and asked: "{query}"

Based on our knowledge base about crop diseases:

{text_context}

Please provide:
1. What disease or condition this might be based on the context
2. Key symptoms to look for
3. Recommended treatment or action steps

Be specific and helpful."""

            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "matched_sources": list(set(matched_sources)),
                "related_images": image_urls,
                "confidence": confidence
            }
            
        except Exception as e:
            raise Exception(f"Error processing image query: {str(e)}")


# Singleton instance
clip_ingest_service = ClipIngestService()
