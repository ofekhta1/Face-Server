from minio import Minio
from typing import BinaryIO, Optional, Tuple
from logging import Logger
import os
from Shared.models.detector_name import DetectorName
from ..base_media_storage import BaseMediaStorage
import json
import mimetypes
import httpx
from minio.error import S3Error
import tempfile

class MinioMediaStorage(BaseMediaStorage):
    def __init__(self, endpoint, access_key, secret_key, logger: Logger):
        self.endpoint = endpoint
        self.logger = logger
        self.default_bucket_name="default_face"
        self.client = Minio(endpoint, access_key, secret_key, secure=False)

    def create_bucket_if_not_exists(self, bucket_name: str):
        """Creates a bucket if it does not already exist and sets public read access."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.logger.info(f"Bucket {bucket_name} does not exist, creating.")
                self.client.make_bucket(bucket_name)
                self._set_public_read_policy(bucket_name)
            else:
                self.logger.info(f"Bucket {bucket_name} already exists.")
        except Exception as e:
            self.logger.error(f"Failed to create bucket {bucket_name}: {str(e)}")

    def init_storage(self, **kwargs):
        self.create_bucket_if_not_exists(self.default_bucket_name)

    def get_bucket_and_path(self, filename: str, detector_name: DetectorName = DetectorName.eran_retinaface, invalid: bool = False) -> Tuple[str, str]:
        """Determine the bucket and object path for a given filename."""
        processed=filename.startswith(("aligned_", "detected_"));
        bucket_name = self.get_bucket(processed,invalid);
        if processed:
            object_name = os.path.join(detector_name, filename)
        else:
            object_name = filename
        return bucket_name, object_name                

    def get_bucket(self,processed:bool,invalid=False):
        self.logger.warning("Using default bucket name");
        return self.default_bucket_name;
    def _set_public_read_policy(self, bucket_name: str):
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


    def load_media_file(self, filename: str, detector_name: DetectorName = "")->tuple[bytearray,str] :
        """Load an image from Minio, decode it using OpenCV, and save to a temporary file."""
        if not self.allowed_file(filename):
            self.logger.error(f"Invalid file type {filename}")
            return None,""

        try:
            bucket_name, object_name = self.get_bucket_and_path(filename, detector_name)
            response = self.client.get_object(bucket_name, object_name)

            with response as data_stream:
                image_data = bytearray()
                for chunk in data_stream.stream(4096):
                    image_data.extend(chunk)

        except Exception as e:
            self.logger.error(f"Error loading image from Minio: {str(e)}")
            return None,""
        
        tmp_file_path = os.path.join('/tmp', os.path.basename(filename))

        try:
            with open(tmp_file_path, 'wb') as temp_file:
                temp_file.write(image_data)
        except Exception as e:
            self.logger.error(f"Error saving image to /tmp: {str(e)}")
            return None, ""

        self.logger.info(f"Image loaded and saved to temporary file: {tmp_file_path}")
        return image_data, tmp_file_path

    def save_media_file(self, image: BinaryIO, filename: str, file_size: int, detector_name: DetectorName = "", invalid: bool = False) -> bool:
        """Save a file to Minio."""
        try:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or 'application/octet-stream'
            bucket_name, object_name = self.get_bucket_and_path(filename, detector_name, invalid)
            self.client.put_object(bucket_name, object_name, image, content_type=content_type, length=file_size)
        except Exception as e:
            self.logger.error(f"Error saving image to Minio: {str(e)}")
            return False

        self.logger.info(f"Uploaded file {filename} to Minio")
        return True

    def fsave_media_file(self, path: str, filename: str, detector_name: Optional[DetectorName] = "", invalid: bool = False) -> bool:
        """Save an image to Minio from a file path."""
        try:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or 'application/octet-stream'
            bucket_name, object_name = self.get_bucket_and_path(filename, detector_name, invalid)
            self.client.fput_object(bucket_name, object_name, path, content_type=content_type)
        except Exception as e:
            self.logger.error(f"Error saving file to Minio: {str(e)}")
            return False

        self.logger.info(f"Uploaded file {filename} to Minio")
        return True

    async def serve_media_file(self, path: str, processed: bool):
        bucket_name = self.get_bucket(processed);

        try:
            url = f"http://{self.endpoint}/{bucket_name}/{path}"
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", url) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except S3Error as e:
            if e.code == 'NoSuchKey':
                yield None
            else:
                self.logger.error(f"Error serving image from Minio: {str(e)}")
                yield None


    def remove_media_file(self, filename: str, detector_name: DetectorName = "", invalid: bool = False) -> bool:
        """Remove an image from Minio."""
        try:
            bucket_name, object_name = self.get_bucket_and_path(filename, detector_name, invalid)
            self.client.remove_object(bucket_name, object_name)
            return True
        except Exception as e:
            self.logger.error(f"Error removing image from Minio: {str(e)}")
            return False

    def file_exists(self, filename: str, detector_name: DetectorName, invalid: bool) -> bool:
        """Check if an image file exists in Minio."""
        try:
            bucket_name, object_name = self.get_bucket_and_path(filename, detector_name, invalid)
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            else:
                self.logger.error(f"Error checking if file {filename} exists in Minio: {str(e)}")
                return False
            
    def download(self, filename, detector_name, invalid:bool)->str|None:
        """Load an image from Minio, and save to a temporary file."""
        _,path=self.load_media_file(filename,detector_name)
        return path