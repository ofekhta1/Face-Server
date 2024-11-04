from minio import Minio
from typing import BinaryIO, Optional
from logging import Logger
import os
import numpy as np
from models.detector_name import DetectorName
import cv2
from .base_image_storage import BaseImageStorage
import json
import mimetypes
import httpx
from minio.error import S3Error
import tempfile

class MinioImageStorage(BaseImageStorage):
    def __init__(self, endpoint, access_key, secret_key, logger: Logger):
        self.endpoint = endpoint
        self.logger = logger
        self.pool_bucket_name = "pool"
        self.processed_bucket_name = "processed"
        self.invalid_bucket_name = "noface"
        self.client = Minio(endpoint, access_key, secret_key, secure=False)

    def __create_bucket_if_not_exists(self, bucket_name: str):
        """Creates a bucket if it does not already exist and sets public read access."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.logger.info(f"Bucket {bucket_name} does not exist, creating.")
                self.client.make_bucket(bucket_name)
                self.__set_public_read_policy(bucket_name)
            else:
                self.logger.info(f"Bucket {bucket_name} already exists.")
        except Exception as e:
            self.logger.error(f"Failed to create bucket {bucket_name}: {str(e)}")

    def __set_public_read_policy(self, bucket_name: str):
        """Set the public-read policy for the bucket to allow anonymous access."""
        try:
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket_name}/*"
                    }
                ]
            }
            policy_json = json.dumps(policy)
            self.client.set_bucket_policy(bucket_name, policy_json)
            self.logger.info(f"Public-read policy applied to bucket {bucket_name}.")
        except Exception as e:
            self.logger.error(f"Failed to set public-read policy for bucket {bucket_name}: {str(e)}")

    def init_storage(self, **kwargs):
        """Initialize storage by ensuring required buckets exist with public-read access."""
        self.__create_bucket_if_not_exists(self.pool_bucket_name)
        self.__create_bucket_if_not_exists(self.processed_bucket_name)
        self.__create_bucket_if_not_exists(self.invalid_bucket_name)


    def load_image(self, filename: str, detector_name: DetectorName = "") -> tuple[cv2.typing.MatLike, str]:
        """Load an image from Minio, decode it using OpenCV, and save to a temporary file."""
        if not self.allowed_file(filename):
            self.logger.error(f"Invalid file type {filename}")
            return None

        try:
            # Determine which bucket and path to use
            if filename.startswith("aligned_") or filename.startswith("detected_"):
                save_path = os.path.join(detector_name, filename)
                response = self.client.get_object(self.processed_bucket_name, save_path)
            else:
                response = self.client.get_object(self.pool_bucket_name, filename)

            # Read the image data from Minio
            with response as data_stream:
                image_data = bytearray()
                for chunk in data_stream.stream(4096):
                    image_data.extend(chunk)

        except Exception as e:
            self.logger.error(f"Error loading image from Minio: {str(e)}")
            return None

        # Convert image data to NumPy array and decode it using OpenCV
        image_np = np.asarray(image_data, dtype="uint8")
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        _, file_extension = os.path.splitext(filename)
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            # Write the original image bytes to the temp file
            temp_file.write(image_data)

        self.logger.info(f"Image loaded and saved to temporary file: {temp_file_path}")
        
        # Return the OpenCV image and the path to the temporary file
        return img, temp_file_path
    

    def save_image(self, image:BinaryIO, filename: str,file_size:int, detector_name: DetectorName="",invalid:bool=False)->bool:
        """Save an image to Minio."""
        try:
            content_type, _ = mimetypes.guess_type(filename)  # Infer content type
            content_type = content_type or 'application/octet-stream'  # Default if none found

            if invalid:
                self.client.put_object(self.invalid_bucket_name, filename, image,content_type=content_type,length=file_size)
            elif filename.startswith("aligned_") or filename.startswith("detected_"):
                save_path = os.path.join(detector_name, filename)
                self.client.put_object(self.processed_bucket_name, save_path, image,content_type=content_type,length=file_size)
            else:
                self.client.put_object(self.pool_bucket_name, filename, image,content_type=content_type,length=file_size)
        except Exception as e:
            self.logger.error(f"Error saving image to Minio: {str(e)}")
            return False

        self.logger.info(f"Uploaded file {filename} to Minio")
        return True

    def fsave_image(self, path: str, filename: str, detector_name: Optional[DetectorName] = "", invalid: bool = False) -> bool:
        """Save an image to Minio from a file path."""
        try:
            content_type, _ = mimetypes.guess_type(filename)  # Infer content type
            content_type = content_type or 'application/octet-stream'  # Default if none found

            if invalid:
                self.client.fput_object(self.invalid_bucket_name, filename, path,content_type=content_type)
            elif filename.startswith("aligned_") or filename.startswith("detected_"):
                save_path = os.path.join(detector_name, filename)
                self.client.fput_object(self.processed_bucket_name, save_path, path,content_type=content_type)
            else:
                self.client.fput_object(self.pool_bucket_name, filename, path,content_type=content_type)
        except Exception as e:
            self.logger.error(f"Error saving image to Minio: {str(e)}")
            return False

        self.logger.info(f"Uploaded file {filename} to Minio")
        return True

    async def serve_image(self,path:str,processed:bool):
        bucket_name = self.processed_bucket_name if processed else self.pool_bucket_name

        try:
            url=f"http://{self.endpoint}/{bucket_name}/{path}"
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", url) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except S3Error as e:
            # Handle file not found or other errors
            if e.code == 'NoSuchKey':
                yield None
            else:
                self.logger.error(f"Error serving image from Minio: {str(e)}")
                yield None