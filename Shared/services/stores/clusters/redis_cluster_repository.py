import pickle
import os
import sys

import redis.asyncio
import redis.asyncio.client
from Shared.models.embedder_name import EmbedderName
from Shared.models.detector_name import DetectorName
sys.path.append(os.path.abspath(".."))
from Shared.models.stored_group import StoredDetectorGroup, StoredGroup
from ...models.model_loader import ModelLoader
from ...logging import ConsoleLogger
from .base_cluster_repository import BaseClusterRepository
import redis
from typing import Optional
class RedisClusterRepository(BaseClusterRepository):
    def __init__(self, host: str, port: int,logger:ConsoleLogger):
        self.client = redis.asyncio.client.Redis(host=host, port=port, db=0,decode_responses=True)
        self.logger=logger
    
    @staticmethod
    def _get_index_key( detector_name: DetectorName, embedder_name: EmbedderName) -> str:
        return f"{detector_name}_{embedder_name}:index"
    @staticmethod
    def _get_group_key( detector_name: DetectorName, embedder_name: EmbedderName) -> str:
        return f"{detector_name}_{embedder_name}:group_id"

    async def has_group(self, detector_name:DetectorName, embedder_name:EmbedderName):
        group_key = self._get_index_key(detector_name, embedder_name)
        return await self.client.exists(group_key)
    
    async def get_group_id(self,name:str, detector_name:DetectorName, embedder_name:EmbedderName)->str:
        key=self._get_index_key(detector_name,embedder_name);
        id=await self.client.hget(key,name)
        return id
    
    async def save_index(
        self,
        data: dict[str, str],
        id_groups: dict[str, list[str]],
        core_faces: dict,
        detector_name: str,
        embedder_name: str,
    ):
        index_key = self._get_index_key(detector_name, embedder_name)
        group_key_prefix = self._get_group_key(detector_name, embedder_name)
        
        # First pipeline to get all keys matching the prefix
        scan_pipeline = self.client.pipeline()
        # SCAN command with match pattern and COUNT hint
        scan_pipeline.scan(match=f"{group_key_prefix}:*", count=1000)
        keys_to_delete = await scan_pipeline.execute()
        # SCAN returns a tuple of (cursor, [keys])
        existing_group_keys = keys_to_delete[0][1]
        
        # Start main pipeline for all operations
        pipeline = self.client.pipeline()
        pipeline.multi()
        
        # Delete all existing group keys if any were found
        if existing_group_keys:
            pipeline.delete(*existing_group_keys)
        
        for image, group_id in data.items():
            pipeline.hset(index_key, image, group_id)
        
        for group_id, images in id_groups.items():
            group_key = group_key_prefix + f":{group_id}"
            pipeline.rpush(group_key, *images)
        
        # Execute the pipeline
        await pipeline.execute()
        return id_groups


    async def load_index(self, detector_name):
        pass;

    async def delete_index(self, detector_name):
        pass;
    
    async def get_by_id(
        self, detector_name: DetectorName, embedder_name: EmbedderName, group_id: str
    ) -> list[str]:
        
        key=self._get_group_key(detector_name, embedder_name)+f":{group_id}"
        return [img for img in await self.client.lrange(key, 0, -1)]


    async def change_group_name(
        self, old_id: str, new_id: str, detector_name: str, embedder_name: str
    ):
        faces = await self.get_by_id(detector_name, embedder_name, old_id)
        index_key = self._get_index_key(detector_name, embedder_name)
        old_group_key = self._get_group_key(detector_name, embedder_name) + f":{old_id}"
        new_group_key = self._get_group_key(detector_name, embedder_name) + f":{new_id}"
    
        # Start a pipeline
        pipeline=self.client.pipeline()
        pipeline.multi()
        for face in faces:
            pipeline.hset(index_key, face, new_id)
            # Remove the image from the old group's list
            pipeline.lrem(old_group_key, 0, face)
            
            # Add the image to the new group's list
            pipeline.rpush(new_group_key, face)

        # Execute the pipeline
        await pipeline.execute()
    
        return faces
    async def get_all_faces(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> dict[str, str]:
        key=self._get_index_key(detector_name, embedder_name)
        result=await self.client.hgetall(key)
        if result is None:
            return result
        image_group_dict = {image: group for image, group in result.items()}
        return image_group_dict 

    async def get_id_groups(self, detector_name: DetectorName, embedder_name: EmbedderName):
        # Get all group keys for the detector
        key = self._get_group_key(detector_name, embedder_name)+":*"  
        group_keys =await self.client.keys(key)  # Find all group keys for the detector
        sorted_keys = sorted(group_keys) # Sort
        group_images = {}
        
        for key in sorted_keys:
            group_id = key.split(":")[-1]  # Extract group ID from the key
            images = [img for img in await self.client.lrange(key, 0, -1)]  # Get all images in the group
            group_images[group_id] = images  # Store group ID and its images in the dictionary
        
        return group_images


    async def get_thumbnails(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> Optional[dict[str, str]]:
        key = self._get_group_key(detector_name, embedder_name) + ":*"
        group_keys =await self.client.keys(key)
        if not group_keys:
            self.logger.debug(f"No thumbnails found for {detector_name}, {embedder_name}.")
            return None

        first_images = {}
        pipeline=self.client.pipeline()
        pipeline.multi()
        for key in group_keys:
            pipeline.lindex(key, 0)  # Get the first image in each group
        responses =await pipeline.execute()
        
        for idx, key in enumerate(group_keys):
            group_id = key.split(":")[-1]
            first_images[group_id] = responses[idx] if responses[idx] else None
        
        return first_images

    async def assign(
        self,
        img_name: str,
        group_id: str,
        detector_name: DetectorName,
        embedder_name: EmbedderName,
    ) -> list[str]:
        if not await self.has_group(detector_name, embedder_name):
            self.logger.warning(f"Group for detector {detector_name} and embedder {embedder_name} does not exist.")
            return False
        
        index_key = self._get_index_key(detector_name, embedder_name)
        old_id = await self.client.hget(index_key, img_name)
        if old_id is not None:
            old_group_key = self._get_group_key(detector_name, embedder_name) + f":{old_id}"
            new_group_key = self._get_group_key(detector_name, embedder_name) + f":{group_id}"
    
            # Start a pipeline
            pipeline=self.client.pipeline()
            pipeline.multi()
            pipeline.hset(index_key, img_name, group_id)
            # Remove the image from the old group's list
            pipeline.lrem(old_group_key, 0, img_name)
            
            # Add the image to the new group's list
            pipeline.rpush(new_group_key, img_name)

            # Execute the pipeline
            await pipeline.execute()
    
            self.logger.info(f"Image {img_name} reassigned from {old_id} to {group_id}.")
    
        return await self.get_by_id(detector_name, embedder_name, group_id)