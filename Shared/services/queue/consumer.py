import threading
import traceback
from Shared.services.util import face_path
import asyncio
from Shared.models.processing_message import ProcessingMessage
from Shared.models.detector_name import DetectorName
from Shared.models.embedder_name import EmbedderName
from Shared.services.stores.jobs import BaseJobManager
from Shared.services.stores.media.base_image_storage import BaseImageStorage
from Shared.services.processing.local.local_image_processor import LocalImageProcessor
from Shared.services.processing.triton.triton_image_processor import TritonImageProcessor
from Shared.services.stores.embeddings import BaseImageEmbeddingManager
from Shared.services.queue import RabbitMQQueue,InMemoryProcessingQueue
from Shared.models.errors.base_error import BaseError
from Shared.models.job_status import JobStatus
import os
import numpy as np
import Shared.services.util as util
from queue import Empty as EmptyQueue
import threading
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Optional

class Consumer:
    def __init__(self, emb_manager: BaseImageEmbeddingManager,
                 image_processor: LocalImageProcessor | TritonImageProcessor,
                 queue: InMemoryProcessingQueue | RabbitMQQueue,
                 job_manager: BaseJobManager,
                 image_storage: BaseImageStorage,
                 URL: str = ""):
        self.image_processor = image_processor
        self.image_storage = image_storage
        self.emb_manager = emb_manager
        self.job_manager = job_manager
        self.URL = URL
        self.queue = queue
        self.processing_lock = threading.Lock()
        
        # Create a queue for passing messages between threads
        self.message_queue = Queue()
        
        # Store the event loop or create a new one if needed
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def start_consuming(self):
        # Start the consumer thread
        self.thread = threading.Thread(target=self._consumer_thread)
        self.thread.daemon = True
        self.thread.start()
        
        # Start the async message processor
        self.loop.create_task(self._process_message_queue())

    def _consumer_thread(self):
        """Thread that receives messages from the queue and passes them to the async processor"""
        try:
            self.queue.start_consuming(self._handle_queue_message)
        except Exception as e:
            print(f"Consumer thread error: {str(e)}")
            traceback.print_exc()

    def _handle_queue_message(self, message: ProcessingMessage, delivery_tag):
        """Callback function that puts messages into our internal queue"""
        self.message_queue.put((message, delivery_tag))
        

    async def _process_message_queue(self):
        """Async task that processes messages from our internal queue"""
        while True:
            try:
                # Use executor to avoid blocking the event loop when getting messages
                message, delivery_tag = await self.loop.run_in_executor(
                    None, self.message_queue.get, True, 1.0
                )
                await self.job_manager.update_job_status(message.id,job_status=JobStatus.processing)        
                if self.processing_lock.acquire(blocking=False):
                    try:
                        await self.handle_task(message)
                    except Exception as e:
                        print(f"Error processing task: {str(e)}")
                        traceback.print_exc()
                    finally:
                        self.processing_lock.release()
                        
            except EmptyQueue:
                # Queue timeout, continue polling
                continue
            except Exception as e:
                print(f"Error in message queue processor: {str(e)}")
                traceback.print_exc()
                self.queue.fail_message(delivery_tag,message)
                await asyncio.sleep(1)  # Avoid tight loop on persistent errors

    async def handle_task(self, message: ProcessingMessage):
        """Handle individual processing tasks"""
        print(f"Processing Images: {message.data_paths}")
        invalid_images=None
        try:
            result = await self.process_images(
                file_names=message.data_paths,
                return_detector=message.return_detector,
                return_embedder=message.return_embedder,
                save_invalid=message.save_invalid
            )
            
            if isinstance(result, BaseError):
                await self.job_manager.fail_job(message.id, str(result.reason),invalid_images=message.data_paths)
                raise Exception(result.reason)
            else:
                valid_images, invalid_images = result
                await self.job_manager.complete_job(message.id, valid_images, invalid_images)
                
        except Exception as e:
            print(f"Task handling error: {str(e)}")
            traceback.print_exc()
            if invalid_images is None:
                invalid_images = message.data_paths
                valid_images=[]
            await self.job_manager.fail_job(message.id, str(e),invalid_images,valid_images)

    async def process_images(self, file_names: list[str],
                           return_detector: DetectorName,
                           return_embedder: EmbedderName,
                           save_invalid: bool) -> tuple[list[str], list[str]] | BaseError:
        faces_length = []
        detector_indices: list[dict[str, list[int]]] = [{}] * len(file_names)
        valid_images = []
        generated_embeddings: list[dict[str, list[np.ndarray]]] = {}
        invalid_images = []
        
        for i, filename in enumerate(file_names):
            img, temp_file_path = self.image_storage.load_image(filename)
            if img is None:
                invalid_images.append(filename)
                continue
            temp_dir = os.path.dirname(temp_file_path)
            faces_dir = os.path.join(temp_dir, os.path.basename(temp_file_path) + '_faces')
            os.makedirs(faces_dir, exist_ok=True)

            generated_embeddings,metadata, errors = await self.image_processor.process_image(
                temp_file_path, filename, faces_dir
            )

            return_key = f"{return_detector}_{return_embedder}"
            if return_key not in generated_embeddings or len(generated_embeddings[return_key]) == 0:
                faces_length.append(0)
                if save_invalid:
                    self.image_storage.fsave_media_file(temp_file_path, filename, invalid=True)
                os.remove(temp_file_path)
                invalid_images.append("no_face/" + filename)
                continue

            valid_images.append(filename)
            embeddings = generated_embeddings[return_key]
            faces_length.append(len(embeddings))

            seen_detectors = []
            for models in generated_embeddings:
                detector, _ = models.split('_')
                if detector not in seen_detectors:
                    seen_detectors.append(detector)
                    count = len(generated_embeddings[models])
                    for face_num in range(count):
                        aligned_filename = face_path(filename, face_num)
                        path = os.path.join(faces_dir, detector, aligned_filename)
                        self.image_storage.fsave_media_file(path, aligned_filename, detector)

            self.emb_manager.save()

        return valid_images, invalid_images