"""
PDF to Pinecone Ingestion Pipeline
Extracts text and images from PDFs, generates embeddings, and loads into Pinecone
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# PDF and Image Processing
import fitz  # PyMuPDF
from PIL import Image
import io

# Text Processing and Embeddings
from sentence_transformers import SentenceTransformer
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
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
    PINECONE_INDEX_NAME = "pdf-documents"
    PINECONE_DIMENSION = 384  # for all-MiniLM-L6-v2
    PINECONE_METRIC = "cosine"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"
    
    # Text processing
    CHUNK_SIZE = 500  # tokens
    CHUNK_OVERLAP = 50  # tokens
    
    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Image settings
    EXTRACT_IMAGES = True
    MIN_IMAGE_SIZE = (100, 100)  # Minimum width, height


class PDFExtractor:
    """Extract text and images from PDF files"""
    
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.IMAGE_DIR, exist_ok=True)
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and images from a single PDF"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        pdf_name = Path(pdf_path).stem
        
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Extract images
            images = []
            if self.config.EXTRACT_IMAGES:
                images = self._extract_images_from_page(page, pdf_name, page_num)
            
            pages_data.append({
                'page_number': page_num + 1,
                'text': text,
                'images': images
            })
        
        doc.close()
        
        return {
            'pdf_name': pdf_name,
            'total_pages': len(doc),
            'pages': pages_data
        }
    
    def _extract_images_from_page(self, page, pdf_name: str, page_num: int) -> List[str]:
        """Extract images from a PDF page"""
        image_list = page.get_images()
        extracted_images = []
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Open image to check size
            image = Image.open(io.BytesIO(image_bytes))
            
            # Filter out small images (likely logos/icons)
            if image.size[0] >= self.config.MIN_IMAGE_SIZE[0] and \
               image.size[1] >= self.config.MIN_IMAGE_SIZE[1]:
                
                # Save image
                image_filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.png"
                image_path = os.path.join(self.config.IMAGE_DIR, image_filename)
                image.save(image_path)
                extracted_images.append(image_path)
                logger.info(f"Extracted image: {image_filename}")
        
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
        # Remove special characters if needed
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
        chunks = self.chunk_text(cleaned)
        return chunks


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


class PineconeLoader:
    """Load data into Pinecone"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self._get_or_create_index()
    
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
        else:
            logger.info(f"Using existing Pinecone index: {index_name}")
        
        return self.pc.Index(index_name)
    
    def upsert_chunks(self, chunks_data: List[Dict[str, Any]], batch_size: int = 100):
        """Upsert text chunks with embeddings to Pinecone"""
        vectors = []
        
        for i, chunk_data in enumerate(chunks_data):
            vector_id = chunk_data['id']
            embedding = chunk_data['embedding']
            metadata = chunk_data['metadata']
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
            
            # Upsert in batches
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                logger.info(f"Upserted batch of {len(vectors)} vectors")
                vectors = []
        
        # Upsert remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)
            logger.info(f"Upserted final batch of {len(vectors)} vectors")
    
    def get_index_stats(self):
        """Get statistics about the index"""
        stats = self.index.describe_index_stats()
        return stats


class PDFIngestionPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize components
        self.pdf_extractor = PDFExtractor(self.config)
        self.text_processor = TextProcessor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.pinecone_loader = PineconeLoader(self.config)
        
        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
    
    def process_pdf(self, pdf_path: str) -> int:
        """Process a single PDF and load to Pinecone"""
        # Step 1: Extract content
        pdf_data = self.pdf_extractor.extract_from_pdf(pdf_path)
        pdf_name = pdf_data['pdf_name']
        
        # Step 2: Process and chunk text
        all_chunks_data = []
        chunk_id_counter = 0
        
        for page_data in pdf_data['pages']:
            page_num = page_data['page_number']
            text = page_data['text']
            images = page_data['images']
            
            # Chunk text
            chunks = self.text_processor.process_page_text(text)
            
            # Generate embeddings for chunks
            if chunks:
                embeddings = self.embedding_generator.generate_embeddings(chunks)
                
                for chunk_idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_data = {
                        'id': f"{pdf_name}_page{page_num}_chunk{chunk_idx}",
                        'embedding': embedding,
                        'metadata': {
                            'pdf_name': pdf_name,
                            'page_number': page_num,
                            'chunk_index': chunk_idx,
                            'text': chunk_text,
                            'has_images': len(images) > 0,
                            'image_paths': json.dumps(images) if images else None
                        }
                    }
                    all_chunks_data.append(chunk_data)
                    chunk_id_counter += 1
        
        # Step 3: Load to Pinecone
        if all_chunks_data:
            self.pinecone_loader.upsert_chunks(all_chunks_data)
            logger.info(f"Successfully processed {pdf_name}: {chunk_id_counter} chunks")
        else:
            logger.warning(f"No chunks generated for {pdf_name}")
        
        # Save processing summary
        summary = {
            'pdf_name': pdf_name,
            'total_pages': pdf_data['total_pages'],
            'total_chunks': chunk_id_counter,
            'images_extracted': sum(len(p['images']) for p in pdf_data['pages'])
        }
        
        summary_path = os.path.join(self.config.OUTPUT_DIR, f"{pdf_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return chunk_id_counter
    
    def process_directory(self, pdf_dir: str = None):
        """Process all PDFs in a directory"""
        pdf_dir = pdf_dir or self.config.PDF_DIR
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        total_chunks = 0
        for pdf_file in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_file))
                total_chunks += chunks
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        logger.info(f"Pipeline complete! Total chunks processed: {total_chunks}")
        
        # Show index stats
        stats = self.pinecone_loader.get_index_stats()
        logger.info(f"Pinecone index stats: {stats}")


def main():
    """Main entry point"""
    # Create necessary directories
    os.makedirs(Config.PDF_DIR, exist_ok=True)
    os.makedirs(Config.IMAGE_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = PDFIngestionPipeline()
    
    # Process all PDFs in the directory
    pipeline.process_directory()
    
    # Or process a single PDF
    # pipeline.process_pdf("data/pdfs/your-document.pdf")


if __name__ == "__main__":
    main()