from services.models.model_loader import ModelLoader
from services.processing.image_loader import ImageLoader
from models.errors.base_error import BaseError
import numpy as np
from services.processing.local.local_embedding_generator import LocalEmbeddingGenerator
from services.stores import InMemoryImageEmbeddingManager,MilvusImageEmbeddingManager
from services.processing.face_aligner import FaceAligner
from models.detector_name import DetectorName

class LocalImageProcessor:
    def __init__(self,emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager,
                 face_aligner:FaceAligner,embedding_generator:LocalEmbeddingGenerator,model_loader:ModelLoader):
        self.model_loader=model_loader
        self.emb_manager=emb_manager
        self.face_aligner=face_aligner
        self.embedding_generator=embedding_generator
        

    async def process_image(self,file_path:str,save_file_name:str,faces_dir:str)->tuple[dict[str,list[np.ndarray]],list[str]]:
        """
        Processes the given image file to generate face embeddings using multiple detectors and embedders.

        This function:
        1. Loads the image from the provided path.
        2. Iterates through all available face detection models (detectors) to detect faces in the image.
        3. For each detected face, aligns the face,saves it to storage and generates an embedding for it with each embedder.
        4. Uses a gender and age model to extract more metadata.
        5. Stores the generated embeddings in a dictionary, keyed by the combination of detector and embedder names.

        Args:
            file_path (str): The path to the image file to be processed.

        Returns:
            tuple: A dictionary containing the generated embeddings for each detector and embedder combination, 
                and a list of errors encountered during processing.
        """
        errors=[]
        generated_embeddings: dict[str, list[np.ndarray]] = {}
        # Load the imageQ
        img=ImageLoader.load_image(file_path);
        gender_age_model=self.model_loader.load_genderage("MobileNetCeleb0.25_CelebA");
        
        # load model
        for detector_name in self.model_loader.model_registry["detectors"]:
            detector = self.model_loader.load_detector(model_name=detector_name)
            
            det_result = self.face_aligner.create_aligned_images(
                    file_path,save_file_name, detector,img,faces_dir)
            
            if isinstance(det_result,BaseError):
                    errors.append(det_result)
                    continue;
            img, faces = det_result
            genders,ages=gender_age_model.get_gender_age(img,faces)
            for embedder_name in self.model_loader.model_registry["embedders"]:
                embedder = self.model_loader.load_embedder(
                    model_name=embedder_name
                )
                # Create cropped images for all faces detected and store them in the respective model folder under static/{model}/

                # Generate the embeddings for all faces and store them for future indexing
                result = self.embedding_generator.generate_all_emb(
                    img,
                    faces,
                    faces_dir,
                    detector,
                    embedder,
                    filename=save_file_name
                )
                if isinstance(result,BaseError):
                    errors.append(result)
                    continue;
                img, face_embeddings,faces=result


                if face_embeddings is None:
                    continue

                generated_embeddings[f"{detector_name}_{embedder_name}"] = (
                    np.array([e.embedding for e in face_embeddings])
                )

                self.emb_manager.add_embedding_typed(face_embeddings,detector_name,embedder_name)


        return generated_embeddings,errors