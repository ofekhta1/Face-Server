import threading
import traceback
import asyncio
from models.processing_message import ProcessingMessage
from .in_memory_processing_queue import InMemoryProcessingQueue
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName
from services.processing.local.local_image_processor import LocalImageProcessor
from services.processing.triton.triton_image_processor import TritonImageProcessor
from services.stores import InMemoryImageEmbeddingManager,MilvusImageEmbeddingManager
from models.errors.base_error import BaseError
from config.app_paths import AppPaths
import os
import numpy as np
import services.util as util

class Consumer:
    def __init__(self, queue: InMemoryProcessingQueue,emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager,
                 image_processor:LocalImageProcessor|TritonImageProcessor,URL=""):
        self.image_processor = image_processor
        self.emb_manager = emb_manager
        self.queue = queue
        self.URL = URL
        self.thread = threading.Thread(target=self.start_event_loop)
        self.thread.daemon = True  # Daemon thread will automatically close with the app
        self.thread.start()
        
    def start_event_loop(self):
        asyncio.run(self.process_queue());
    async def process_queue(self):
        while True:
            task = self.queue.wait_for_task()  # Waits for a task
            if task is None:  # Stop processing if a `None` task is encountered
                break
            try:
                await self.handle_task(task)
                self.queue.complete_task()
            except Exception as e:
                print(f"Error processing task: {str(e)}")
                traceback.print_exc()

                continue

    async def handle_task(self, message:ProcessingMessage):
        print(f"Processing Images: {message.image_paths}")
        await self.process_images(file_names=message.image_paths,return_detector=message.return_detector,
                            return_embedder=message.return_embedder,save_invalid=message.save_invalid)



    async def process_images(self,file_names:list[str],return_detector:DetectorName,return_embedder:EmbedderName,save_invalid:bool)->bool|BaseError:
        faces_length = []
        detector_indices: list[dict[str, list[int]]] = [{}]*len(file_names)
        valid_images = []
        generated_embeddings: list[dict[str, list[np.ndarray]]] = {}
        # get request parameters    gender_age_model = ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA")


        # true if images will be saved without containing faces
        i = -1
        invalid_images = []
        for file in file_names:
            i += 1
            path = os.path.join(AppPaths.UPLOAD_FOLDER, file)
            # load model
            generated_embeddings,errors=await self.image_processor.process_image(file)
            return_key=f"{return_detector}_{return_embedder}"
            if return_key not in generated_embeddings or len(generated_embeddings[return_key])==0:
                faces_length.append(0)
                # if images with no detected faces are allowed save them under the no face directory
                if save_invalid:
                    os.replace(
                        path,
                        os.path.join(
                            AppPaths.UPLOAD_FOLDER, "no_face", file
                        ),
                    )
                else:
                    os.remove(path)
                invalid_images.append("no_face/" + file)
            else:
                valid_images.append(file)
                embeddings=generated_embeddings[return_key]
                faces_length.append(len(embeddings))
            # save the current database state
            self.emb_manager.save()
            indices_result= util.get_all_detectors_faces(
                generated_embeddings, return_detector,self.image_processor.model_loader
            )
            if isinstance(indices_result,BaseError):
                return indices_result
            detector_indices[i] =indices_result
