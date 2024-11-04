import threading
import traceback
from services.util import face_path
import asyncio
from models.processing_message import ProcessingMessage
from .in_memory_processing_queue import InMemoryProcessingQueue
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName
from services.stores.images.base_image_storage import BaseImageStorage
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
                 image_processor:LocalImageProcessor|TritonImageProcessor,
                 image_storage:BaseImageStorage,URL=""):
        self.image_processor = image_processor
        self.image_storage = image_storage
        self.emb_manager = emb_manager
        self.queue = queue
        self.URL = URL
        self.processing_lock = threading.Lock()
        self.thread = threading.Thread(target=self.start_event_loop)
        self.thread.daemon = True  # Daemon thread will automatically close with the app
        self.thread.start()
        

    def start_event_loop(self):
        # Start consuming messages and pass the callback to handle tasks
        self.queue.start_consuming(self.process_queue)

    def process_queue(self, message, delivery_tag):
        """ Callback function to process each message """
        if self.processing_lock.acquire(blocking=False):
            try:
                asyncio.run(self.handle_task(message))
            except Exception as e:
                print(f"Error processing task: {str(e)}")
                traceback.print_exc()
            finally:
                self.queue.complete_task(delivery_tag)  # Acknowledge task completion
                self.processing_lock.release()

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
        for filename in file_names:
            i += 1
            img,temp_file_path=self.image_storage.load_image(filename)
            temp_dir = os.path.dirname(temp_file_path)
            faces_dir=os.path.join(temp_dir,os.path.basename(temp_file_path) + '_faces')
            os.makedirs(faces_dir,exist_ok=True);

            # load model
            generated_embeddings,errors=await self.image_processor.process_image(temp_file_path,filename,faces_dir)

            return_key=f"{return_detector}_{return_embedder}"
            if return_key not in generated_embeddings or len(generated_embeddings[return_key])==0:
                faces_length.append(0)
                # if images with no detected faces are allowed save them under the no face directory
                if save_invalid:
                    self.image_storage.fsave_image(temp_file_path,filename,invalid=True)

                os.remove(temp_file_path)
                invalid_images.append("no_face/" + filename)
                continue;
            else:
                valid_images.append(filename)
                embeddings=generated_embeddings[return_key]
                faces_length.append(len(embeddings))
            
            # self.image_storage.fsave_image(temp_file_path,filename)
            seen_detectors=[];
            for models in generated_embeddings:
                detector,_=models.split('_')
                if detector not in seen_detectors:
                    seen_detectors.append(detector)
                    count=len(generated_embeddings[models])
                    for face_num in range(count):
                        aligned_filename=face_path(filename,face_num)
                        path=os.path.join(faces_dir,detector,aligned_filename)
                        self.image_storage.fsave_image(path,aligned_filename,detector);

            # save the current database state
            self.emb_manager.save()
            indices_result= util.get_all_detectors_faces(
                generated_embeddings, return_detector,self.image_processor.model_loader
            )
            if isinstance(indices_result,BaseError):
                return indices_result
            detector_indices[i] =indices_result
