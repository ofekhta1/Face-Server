from models.file_video_input import FileVideoInput
import subprocess
from pathlib import Path
from Shared.services.stores.embeddings.in_memory_image_embedding_manager import InMemoryImageEmbeddingManager
from Shared.services.models.triton_model_loader import TritonModelLoader
from Shared.services.logging import ConsoleLogger
import tempfile
import shutil
import os
import json
class InputHandler:
    sources={}

    def remove_source(self,source_id):
        if source_id in self.sources:
            input:FileVideoInput=self.sources[source_id]
            os.remove(input.source_path)
            os.remove(input.path)
            temp_dir = Path(tempfile.gettempdir())  

            # Ensure the directory exists, is a directory, and is within the temporary directory
            if input.out_path.exists() and input.out_path.is_dir() and temp_dir in input.out_path.parents:
                shutil.rmtree(input.out_path)

            del self.sources[source_id]
    def get_source(self,source_id)->FileVideoInput:
        if source_id in self.sources:
            return self.sources[source_id]
        return None  

    def register_source(self,input:FileVideoInput,source_id:str):
        self.sources[source_id]=input
        

    def change_extension_to_mp4(file_path):
        path = Path(file_path)
        return path.with_suffix(".mp4")  
    

    def get_video_dimensions(video_path):
    # Run ffprobe to get video metadata
        cmd = [
            'ffprobe', 
            '-v', 'error',        # Suppress unnecessary output
            '-print_format', 'json',  # Output in JSON format
            '-show_entries', 'stream=width,height',  # Get width and height
            video_path
        ]
        
        # Execute the command and get the output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = json.loads(result.stdout)
        
        # Extract width and height from the metadata
        width = data['streams'][0]['width']
        height = data['streams'][0]['height']
        
        return (width, height)
    
    def process_input_video(path, processing_shape=(720,1280))->FileVideoInput|None:
        try:
            emb_manager=InMemoryImageEmbeddingManager("",TritonModelLoader(),ConsoleLogger(True))
            
            src=Path(path)
            temp_file = Path(tempfile.mktemp()+"_"+src.stem)
            save_path=temp_file.with_suffix(".mp4")

            subprocess.run(
                ["ffmpeg","-y", "-hwaccel", "cuda", "-i", path, "-crf", "18", "-vf", "format=yuv420p", save_path],
                check=True,  # Ensures an exception is raised if the command fails
                capture_output=True,  # Captures stdout and stderr
                text=True  # Returns output as text instead of bytes
            )
            print(f"Video processed successfully: {save_path}")
            output_dir = temp_file.parent / f"{temp_file.stem}_output"
            output_dir.mkdir(exist_ok=True)  

            original_width,original_height  = InputHandler.get_video_dimensions(save_path)
            resized_height, resized_width = processing_shape[0],processing_shape[1]
            
            # Calculate scaling and padding for both width and height
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            
            # Use the minimum scale to maintain aspect ratio
            scale = min(scale_x, scale_y)
            
            # Calculate new dimensions while maintaining aspect ratio
            new_width = original_width * scale
            new_height = original_height * scale
            
            # Calculate padding to center the image
            padding_x = (resized_width - new_width) / 2
            padding_y = (resized_height - new_height) / 2
            

        
            input=FileVideoInput(src,save_path,output_dir,scale,(padding_x,padding_y));
            input.emb_manager=emb_manager
            return input
        except subprocess.CalledProcessError as e:
            print(f"Error processing video: {e.stderr}")
            return None