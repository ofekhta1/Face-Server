from minio import Minio
from logging import Logger
from ..base_video_storage import BaseVideoStorage
from .minio_media_storage import MinioMediaStorage

class MinioVideoStorage(MinioMediaStorage,BaseVideoStorage):
    def __init__(self, endpoint, access_key, secret_key, logger: Logger):
        self.endpoint = endpoint
        self.logger = logger
        self.pool_bucket_name = "video-pool"
        self.processed_bucket_name = "video-processed"
        self.invalid_bucket_name = "invalid-video"
        self.client = Minio(endpoint, access_key, secret_key, secure=False)


    def get_bucket(self, processed, invalid=False):
        if processed:
            return self.processed_bucket_name
        elif invalid:
            return self.invalid_bucket_name
        else:
            return self.pool_bucket_name;

    def init_storage(self, **kwargs):
        """Initialize storage by ensuring required buckets exist with public-read access."""
        for bucket in [self.pool_bucket_name, self.processed_bucket_name, self.invalid_bucket_name]:
            self.create_bucket_if_not_exists(bucket)

