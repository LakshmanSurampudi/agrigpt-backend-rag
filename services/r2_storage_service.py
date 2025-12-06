import os
import io
import uuid
from typing import Optional
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()


class R2StorageService:
    """Cloudflare R2 Storage Service (S3-compatible)"""
    
    def __init__(self):
        self.client = None
        self.bucket_name = None
        self.public_url = None
        self.initialized = False
        
    def initialize(self):
        """Initialize R2 client"""
        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        self.public_url = os.getenv("R2_PUBLIC_URL", "").rstrip("/")
        
        if not all([account_id, access_key, secret_key, self.bucket_name]):
            print("Warning: R2 credentials not fully configured. Image storage will be disabled.")
            self.initialized = False
            return
        
        # R2 endpoint format
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"}
            )
        )
        
        self.initialized = True
        print(f"R2 Storage initialized: bucket={self.bucket_name}")
    
    def upload_image(self, image_bytes: bytes, filename: str, content_type: str = "image/png") -> Optional[str]:
        """
        Upload image to R2 and return public URL
        
        Args:
            image_bytes: Image data as bytes
            filename: Original filename (used for generating unique key)
            content_type: MIME type of the image
            
        Returns:
            Public URL of the uploaded image, or None if upload failed
        """
        if not self.initialized:
            print("R2 not initialized, skipping image upload")
            return None
        
        try:
            # Generate unique key
            ext = filename.split(".")[-1] if "." in filename else "png"
            unique_key = f"images/{uuid.uuid4().hex}.{ext}"
            
            # Upload to R2
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=unique_key,
                Body=image_bytes,
                ContentType=content_type
            )
            
            # Return public URL
            public_url = f"{self.public_url}/{unique_key}"
            print(f"Uploaded image to R2: {public_url}")
            return public_url
            
        except Exception as e:
            print(f"Error uploading to R2: {str(e)}")
            return None
    
    def delete_image(self, image_url: str) -> bool:
        """
        Delete image from R2
        
        Args:
            image_url: Public URL of the image to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            # Extract key from URL
            key = image_url.replace(f"{self.public_url}/", "")
            
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            print(f"Deleted image from R2: {key}")
            return True
            
        except Exception as e:
            print(f"Error deleting from R2: {str(e)}")
            return False


# Singleton instance
r2_storage = R2StorageService()
