import os
import uuid
from typing import Optional
from pathlib import Path

# Directory for storing images
STATIC_DIR = Path(__file__).parent.parent / "static" / "images"


class LocalStorageService:
    """Local file storage service for images (replaces R2)"""
    
    def __init__(self):
        self.static_dir = STATIC_DIR
        self.initialized = False
        
    def initialize(self):
        """Initialize local storage directory"""
        try:
            # Create directory if it doesn't exist
            self.static_dir.mkdir(parents=True, exist_ok=True)
            self.initialized = True
            print(f"✅ Local storage initialized: {self.static_dir}")
        except Exception as e:
            print(f"Error initializing local storage: {e}")
            self.initialized = False
    
    def upload_image(self, image_bytes: bytes, filename: str, content_type: str = "image/png") -> Optional[str]:
        """
        Save image locally and return relative URL path
        
        Args:
            image_bytes: Image data as bytes
            filename: Original filename
            content_type: MIME type (unused, kept for API compatibility)
            
        Returns:
            Relative URL path to the image (e.g., /static/images/abc123.png)
        """
        if not self.initialized:
            print("Local storage not initialized, initializing now...")
            self.initialize()
        
        try:
            # Generate unique filename
            ext = filename.split(".")[-1] if "." in filename else "png"
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            file_path = self.static_dir / unique_filename
            
            # Write image to file
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            
            # Return relative URL path (will be served by FastAPI static files)
            relative_url = f"/static/images/{unique_filename}"
            print(f"Saved image: {relative_url}")
            return relative_url
            
        except Exception as e:
            print(f"Error saving image locally: {e}")
            return None
    
    def delete_image(self, image_url: str) -> bool:
        """
        Delete image from local storage
        
        Args:
            image_url: Relative URL path of the image
            
        Returns:
            True if deleted successfully
        """
        try:
            # Extract filename from URL
            filename = image_url.split("/")[-1]
            file_path = self.static_dir / filename
            
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted image: {filename}")
                return True
            return False
            
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False
    
    def clear_all(self):
        """Clear all stored images"""
        try:
            for file in self.static_dir.glob("*"):
                file.unlink()
            print("Cleared all stored images")
        except Exception as e:
            print(f"Error clearing images: {e}")


# Singleton instance
local_storage = LocalStorageService()
