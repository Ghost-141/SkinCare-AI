import boto3
from botocore.exceptions import ClientError
from core.config import settings
from core.logger import logger
import os

class S3Client:
    def __init__(self):
        self.bucket = settings.S3_BUCKET
        self.enabled = bool(self.bucket)
        if self.enabled:
            self.client = boto3.client('s3')
            logger.info(f"S3 Client initialized for bucket: {self.bucket}")
        else:
            logger.warning("S3_BUCKET not set. Falling back to local storage.")

    def upload_file(self, local_path: str, s3_key: str):
        """Upload a file to the S3 bucket."""
        if not self.enabled:
            return None
        
        try:
            self.client.upload_file(local_path, self.bucket, s3_key)
            return s3_key
        except Exception as e:
            logger.error(f"S3 Upload Error: {e}")
            return None

    def get_presigned_url(self, s3_key: str, expires_in: int = 3600):
        """Generate a pre-signed URL for a private S3 object."""
        if not self.enabled or not s3_key:
            return None
        
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': s3_key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"S3 Presign Error: {e}")
            return None

s3_client = S3Client()
