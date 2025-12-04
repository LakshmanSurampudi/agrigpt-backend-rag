"""
Multimodal PDF to Pinecone Ingestion Pipeline using CLIP
Extracts text and images from PDFs, generates CLIP embeddings, and loads into Pinecone
Supports cross-modal retrieval: text queries → image results and vice versa
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union
import logging

# PDF and Image Processing
import fitz  # PyMuPDF
from PIL import Image
import io

# CLIP and Embeddings
import torch
from transformers import CLIPProcessor, CLIPModel
import tiktoken

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Configuration for the pipeline"""
    # Directories
    PDF_DIR = "data/pdfs"
    IMAGE_DIR = "data/images"
    OUTPUT_DIR = "data/processed"
    
    # Pinecone settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pinecone-api-key")
    PINECONE_INDEX_NAME = "agribot"
    PINECONE_DIMENSION = 512  # clip-vit-base-patch32 output dimension
    PINECONE_METRIC = "cosine"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"
    
    # Text processing
    CHUNK_SIZE = 300  # tokens - optimized for CLIP
    CHUNK_OVERLAP = 50  # tokens
    
    # CLIP model
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    
    # Image settings
    MIN_IMAGE_SIZE = (100, 100)  # Minimum width, height
    
    # Batch sizes
    EMBEDDING_BATCH_SIZE = 32
    PINECONE_BATCH_SIZE = 100
    
    @classmethod
    def validate(cls):
        """Validate configuration, especially API keys"""
        if not cls.PINECONE_API_KEY or cls.PINECONE_API_KEY == "pinecone-api-key":
            raise ValueError(
                "PINECONE_API_KEY is not set. Please set it as an environment variable:\n"
                "  export PINECONE_API_KEY=your-api-key\n"
                "Or create a .env file with: PINECONE_API_KEY=your-api-key"
            )


class PDFExtractor:
    """Extract text and images from PDF files"""
    
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.IMAGE_DIR, exist_ok=True)
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and images from a single PDF"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pages_data = []
        doc = None
        
        try:
            doc = fitz.open(pdf_path)
            pdf_name = Path(pdf_path).stem
            
            total_pages = len(doc)
            logger.info(f"PDF has {total_pages} pages")
            
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    
                    # Extract text
                    text = page.get_text()
                    
                    # Extract images
                    images = self._extract_images_from_page(page, pdf_name, page_num)
                    
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'images': images
                    })
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    # Continue with next page
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': '',
                        'images': []
                    })
            
            return {
                'pdf_name': pdf_name,
                'pdf_path': pdf_path,
                'total_pages': total_pages,
                'pages': pages_data
            }
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}")
            raise
        finally:
            if doc:
                doc.close()
    
    def _extract_images_from_page(self, page, pdf_name: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a PDF page with metadata"""
        image_list = page.get_images()
        extracted_images = []
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Open image to check size
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Filter out small images (likely logos/icons)
                if image.size[0] >= self.config.MIN_IMAGE_SIZE[0] and \
                   image.size[1] >= self.config.MIN_IMAGE_SIZE[1]:
                    
                    # Save image
                    image_filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.png"
                    image_path = os.path.join(self.config.IMAGE_DIR, image_filename)
                    image.save(image_path)
                    
                    extracted_images.append({
                        'path': image_path,
                        'filename': image_filename,
                        'size': image.size
                    })
                    logger.info(f"Extracted image: {image_filename}")
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num+1}: {e}")
        
        return extracted_images


class TextProcessor:
    """Process and chunk text for embedding"""
    
    def __init__(self, config: Config):
        self.config = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        text = text.strip()
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text.strip():
            return []
        
        # Tokenize text
        tokens = self.encoding.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Get chunk
            end = start + self.config.CHUNK_SIZE
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start += self.config.CHUNK_SIZE - self.config.CHUNK_OVERLAP
        
        return chunks
    
    def process_page_text(self, text: str) -> List[str]:
        """Clean and chunk page text"""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        chunks = self.chunk_text(cleaned)
        return chunks


class CLIPEmbeddingGenerator:
    """Generate CLIP embeddings for text and images"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model: {config.CLIP_MODEL_NAME} on {self.device}")
        
        try:
            self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
            self.model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"Could not load CLIP model {config.CLIP_MODEL_NAME}. Please check your internet connection and try again.") from e
    
    def generate_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate CLIP embeddings for text chunks"""
        if not texts:
            return []
        
        embeddings = []
        batch_size = self.config.EMBEDDING_BATCH_SIZE
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    with torch.no_grad():
                        inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        text_features = self.model.get_text_features(**inputs)
                        # Normalize embeddings
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        embeddings.extend(text_features.cpu().tolist())
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                    # Add zero embeddings for failed batch to maintain alignment
                    for _ in batch:
                        embeddings.append([0.0] * self.config.PINECONE_DIMENSION)
        except Exception as e:
            logger.error(f"Critical error in generate_text_embeddings: {e}")
            raise
        
        return embeddings
    
    def generate_image_embeddings(self, image_paths: List[str]) -> List[Union[List[float], None]]:
        """Generate CLIP embeddings for images
        
        Returns a list of embeddings in the same order as image_paths.
        Returns None for images that failed to load or process.
        """
        if not image_paths:
            return []
        
        # Initialize result list with None for all images
        embeddings = [None] * len(image_paths)
        batch_size = self.config.EMBEDDING_BATCH_SIZE
        
        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                images = []
                successful_indices = []  # Track which images in the batch succeeded
                
                # Load images and track which ones succeeded
                for idx, img_path in enumerate(batch_paths):
                    try:
                        if not os.path.exists(img_path):
                            logger.warning(f"Image file not found: {img_path}")
                            continue
                        
                        img = Image.open(img_path).convert('RGB')
                        images.append(img)
                        successful_indices.append(i + idx)  # Global index in original list
                    except Exception as e:
                        logger.error(f"Error loading image {img_path}: {e}")
                        # Leave as None in embeddings list
                        continue
                
                # Generate embeddings only for successful images
                if images:
                    try:
                        with torch.no_grad():
                            inputs = self.processor(images=images, return_tensors="pt", padding=True)
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            
                            image_features = self.model.get_image_features(**inputs)
                            # Normalize embeddings
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            
                            batch_embeddings = image_features.cpu().tolist()
                            
                            # Map embeddings back to their original positions
                            for result_idx, orig_idx in enumerate(successful_indices):
                                embeddings[orig_idx] = batch_embeddings[result_idx]
                    except Exception as e:
                        logger.error(f"Error generating embeddings for image batch {i//batch_size + 1}: {e}")
                        # Leave failed batch images as None
                        continue
        except Exception as e:
            logger.error(f"Critical error in generate_image_embeddings: {e}")
            raise
        
        return embeddings
    
    def generate_single_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text (for querying)"""
        return self.generate_text_embeddings([text])[0]
    
    def generate_single_image_embedding(self, image_path: str) -> List[float]:
        """Generate embedding for a single image (for querying)"""
        return self.generate_image_embeddings([image_path])[0]


class PineconeLoader:
    """Load data into Pinecone"""
    
    def __init__(self, config: Config):
        self.config = config
        try:
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise RuntimeError("Could not connect to Pinecone. Please check your API key and network connection.") from e
        
        try:
            self.index = self._get_or_create_index()
        except Exception as e:
            logger.error(f"Failed to get or create Pinecone index: {e}")
            raise
    
    def _get_or_create_index(self):
        """Get existing index or create new one"""
        index_name = self.config.PINECONE_INDEX_NAME
        
        # Check if index exists
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=self.config.PINECONE_CLOUD,
                    region=self.config.PINECONE_REGION
                )
            )
            logger.info("Waiting for index to be ready...")
            # Wait for index to be ready (Pinecone needs time to initialize)
            max_wait_time = 60  # Maximum wait time in seconds
            wait_interval = 2  # Check every 2 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                try:
                    index = self.pc.Index(index_name)
                    # Try to get stats to verify index is ready
                    index.describe_index_stats()
                    logger.info(f"Index {index_name} is ready!")
                    return index
                except Exception as e:
                    if "not found" in str(e).lower() or "not ready" in str(e).lower():
                        logger.info(f"Index not ready yet, waiting... ({elapsed_time}s/{max_wait_time}s)")
                        time.sleep(wait_interval)
                        elapsed_time += wait_interval
                    else:
                        # Other error, might be ready but with different error
                        logger.warning(f"Error checking index status: {e}, assuming ready")
                        return self.pc.Index(index_name)
            
            # If we've waited too long, try to return the index anyway
            logger.warning(f"Index creation took longer than {max_wait_time}s, proceeding anyway...")
            return self.pc.Index(index_name)
        else:
            logger.info(f"Using existing Pinecone index: {index_name}")
        
        return self.pc.Index(index_name)
    
    def upsert_vectors(self, vectors_data: List[Dict[str, Any]]):
        """Upsert vectors to Pinecone in batches"""
        if not vectors_data:
            logger.warning("No vectors to upsert")
            return
        
        batch_size = self.config.PINECONE_BATCH_SIZE
        
        for i in range(0, len(vectors_data), batch_size):
            batch = vectors_data[i:i + batch_size]
            
            vectors = []
            for vec_data in batch:
                try:
                    # Validate embedding dimension
                    if len(vec_data['embedding']) != self.config.PINECONE_DIMENSION:
                        logger.error(f"Embedding dimension mismatch for {vec_data['id']}: "
                                   f"expected {self.config.PINECONE_DIMENSION}, "
                                   f"got {len(vec_data['embedding'])}")
                        continue
                    
                    vectors.append({
                        'id': vec_data['id'],
                        'values': vec_data['embedding'],
                        'metadata': vec_data['metadata']
                    })
                except Exception as e:
                    logger.error(f"Error preparing vector {vec_data.get('id', 'unknown')}: {e}")
                    continue
            
            if vectors:
                try:
                    self.index.upsert(vectors=vectors)
                    logger.info(f"Upserted batch: {i+1}-{min(i+batch_size, len(vectors_data))} / {len(vectors_data)}")
                except Exception as e:
                    logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
                    # Continue with next batch
    
    def query(self, query_embedding: List[float], top_k: int = 5, filter_dict: Dict = None):
        """Query the index"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        return results
    
    def get_index_stats(self):
        """Get statistics about the index"""
        return self.index.describe_index_stats()


class MultimodalPDFPipeline:
    """Main pipeline orchestrator for multimodal PDF ingestion"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Validate configuration
        self.config.validate()
        
        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # Initialize components
        try:
            self.pdf_extractor = PDFExtractor(self.config)
            self.text_processor = TextProcessor(self.config)
            logger.info("Initializing CLIP model (this may take a moment)...")
            self.clip_generator = CLIPEmbeddingGenerator(self.config)
            logger.info("Initializing Pinecone connection...")
            self.pinecone_loader = PineconeLoader(self.config)
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def process_pdf(self, pdf_path: str) -> Dict[str, int]:
        """Process a single PDF and load to Pinecone"""
        logger.info(f"Starting pipeline for: {pdf_path}")
        
        # Step 1: Extract content from PDF
        pdf_data = self.pdf_extractor.extract_from_pdf(pdf_path)
        pdf_name = pdf_data['pdf_name']
        
        all_vectors_data = []
        text_chunk_count = 0
        image_count = 0
        
        # Step 2: Process each page
        for page_data in pdf_data['pages']:
            page_num = page_data['page_number']
            page_text = page_data['text']
            page_images = page_data['images']
            
            # Clean page text for context
            cleaned_page_text = self.text_processor.clean_text(page_text)
            
            # Process text chunks
            text_chunks = self.text_processor.process_page_text(page_text)
            
            if text_chunks:
                logger.info(f"Processing {len(text_chunks)} text chunks from page {page_num}")
                text_embeddings = self.clip_generator.generate_text_embeddings(text_chunks)
                
                for chunk_idx, (chunk_text, embedding) in enumerate(zip(text_chunks, text_embeddings)):
                    vector_data = {
                        'id': f"{pdf_name}_page{page_num}_text_chunk{chunk_idx}",
                        'embedding': embedding,
                        'metadata': {
                            'content_type': 'text',
                            'pdf_name': pdf_name,
                            'page_number': page_num,
                            'chunk_index': chunk_idx,
                            'text': chunk_text,
                            'total_pages': pdf_data['total_pages']
                        }
                    }
                    all_vectors_data.append(vector_data)
                    text_chunk_count += 1
            
            # Process images
            if page_images:
                logger.info(f"Processing {len(page_images)} images from page {page_num}")
                image_paths = [img['path'] for img in page_images]
                image_embeddings = self.clip_generator.generate_image_embeddings(image_paths)
                
                for img_idx, (img_data, embedding) in enumerate(zip(page_images, image_embeddings)):
                    # Skip images that failed to generate embeddings
                    if embedding is None:
                        logger.warning(f"Skipping image {img_data['filename']} - failed to generate embedding")
                        continue
                    
                    vector_data = {
                        'id': f"{pdf_name}_page{page_num}_image{img_idx}",
                        'embedding': embedding,
                        'metadata': {
                            'content_type': 'image',
                            'pdf_name': pdf_name,
                            'page_number': page_num,
                            'image_index': img_idx,
                            'image_path': img_data['path'],
                            'image_filename': img_data['filename'],
                            'image_size': json.dumps(img_data['size']),
                            'page_text': cleaned_page_text[:500],  # First 500 chars as context
                            'total_pages': pdf_data['total_pages']
                        }
                    }
                    all_vectors_data.append(vector_data)
                    image_count += 1
        
        # Step 3: Upload to Pinecone
        if all_vectors_data:
            logger.info(f"Uploading {len(all_vectors_data)} vectors to Pinecone...")
            self.pinecone_loader.upsert_vectors(all_vectors_data)
        else:
            logger.warning(f"No vectors generated for {pdf_name}")
        
        # Save processing summary
        summary = {
            'pdf_name': pdf_name,
            'pdf_path': pdf_path,
            'total_pages': pdf_data['total_pages'],
            'text_chunks': text_chunk_count,
            'images': image_count,
            'total_vectors': len(all_vectors_data)
        }
        
        summary_path = os.path.join(self.config.OUTPUT_DIR, f"{pdf_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Processing complete for {pdf_name}")
        logger.info(f"  - Text chunks: {text_chunk_count}")
        logger.info(f"  - Images: {image_count}")
        logger.info(f"  - Total vectors: {len(all_vectors_data)}")
        
        return summary
    
    def query_text(self, query_text: str, top_k: int = 5, content_type: str = None):
        """Query using text and retrieve relevant results"""
        logger.info(f"Querying with text: '{query_text}'")
        
        # Generate embedding for query text
        query_embedding = self.clip_generator.generate_single_text_embedding(query_text)
        
        # Optional filter by content type
        filter_dict = {'content_type': content_type} if content_type else None
        
        # Query Pinecone
        results = self.pinecone_loader.query(query_embedding, top_k=top_k, filter_dict=filter_dict)
        
        return self._format_results(results)
    
    def query_image(self, image_path: str, top_k: int = 5, content_type: str = None):
        """Query using an image and retrieve relevant results"""
        logger.info(f"Querying with image: '{image_path}'")
        
        # Generate embedding for query image
        query_embedding = self.clip_generator.generate_single_image_embedding(image_path)
        
        # Optional filter by content type
        filter_dict = {'content_type': content_type} if content_type else None
        
        # Query Pinecone
        results = self.pinecone_loader.query(query_embedding, top_k=top_k, filter_dict=filter_dict)
        
        return self._format_results(results)
    
    def _format_results(self, results):
        """Format query results for display"""
        formatted = []
        
        for match in results['matches']:
            result = {
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            }
            formatted.append(result)
        
        return formatted


def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multi-modal PDF ingestion into Pinecone")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=os.getenv("PDF_PATH", "data/Pathway-PyData_Global_2022.pdf"),
        help="Path to PDF file to process (default: from PDF_PATH env var or data/Pathway-PyData_Global_2022.pdf)"
    )
    parser.add_argument(
        "--skip-queries",
        action="store_true",
        help="Skip example queries after ingestion"
    )
    args = parser.parse_args()
    
    pdf_path = args.pdf_path
    
    # Create necessary directories
    os.makedirs(Config.PDF_DIR, exist_ok=True)
    os.makedirs(Config.IMAGE_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Validate PDF path
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        logger.info("Please provide a valid PDF path using:")
        logger.info("  - Command line: python multi_modal_data_ingestion2.py --pdf-path <path>")
        logger.info("  - Environment variable: export PDF_PATH=<path>")
        logger.info(f"  - Or place a PDF file at: {pdf_path}")
        return
    
    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {pdf_path}")
        return
    
    try:
        # Initialize pipeline
        logger.info("Initializing multi-modal PDF pipeline...")
        pipeline = MultimodalPDFPipeline()
        
        # Ingest PDF
        logger.info(f"Processing PDF: {pdf_path}")
        summary = pipeline.process_pdf(pdf_path)
        
        print("\n" + "="*60)
        print("INGESTION COMPLETE")
        print("="*60)
        print(json.dumps(summary, indent=2))
        
        if not args.skip_queries:
            # Example queries
            print("\n" + "="*60)
            print("EXAMPLE QUERIES")
            print("="*60)
            
            try:
                # Text query to find images
                print("\n1. Text query (finding images):")
                results = pipeline.query_text("neural network architecture", top_k=3, content_type="image")
                for i, result in enumerate(results, 1):
                    print(f"\n   Result {i} (Score: {result['score']:.4f}):")
                    print(f"   - Type: {result['metadata']['content_type']}")
                    print(f"   - Image: {result['metadata'].get('image_filename', 'N/A')}")
                    print(f"   - Page: {result['metadata']['page_number']}")
            except Exception as e:
                logger.warning(f"Example query 1 failed: {e}")
            
            try:
                # Text query to find text
                print("\n2. Text query (finding text):")
                results = pipeline.query_text("machine learning concepts", top_k=3, content_type="text")
                for i, result in enumerate(results, 1):
                    print(f"\n   Result {i} (Score: {result['score']:.4f}):")
                    print(f"   - Type: {result['metadata']['content_type']}")
                    print(f"   - Text: {result['metadata'].get('text', 'N/A')[:100]}...")
                    print(f"   - Page: {result['metadata']['page_number']}")
            except Exception as e:
                logger.warning(f"Example query 2 failed: {e}")
        
        # Show index stats
        try:
            print("\n" + "="*60)
            print("INDEX STATISTICS")
            print("="*60)
            stats = pipeline.pinecone_loader.get_index_stats()
            print(json.dumps(stats, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to get index stats: {e}")
            
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()