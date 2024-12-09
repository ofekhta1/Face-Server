import cv2
import os
from collections import defaultdict
from typing import Dict, List
from Shared.models.face import Face
from Shared.services.logging.console_logger import ConsoleLogger
from insightface.utils.face_align import norm_crop
import numpy as np

class FrameExtractor:
    def __init__(self, logger: ConsoleLogger):
        self.logger = logger

    def extract_frames_by_identity(self,
                                   video_path: str,
                                   face_groups: Dict[str, List[Face]],
                                   output_directory: str) -> List[str]:
        """
        Extract frames from a video, grouped by identity, and save cropped faces.
        :param video_path: Path to the source video
        :param face_groups: Dictionary mapping group names to lists of faces
        :param output_directory: Directory to save extracted frames and cropped faces
        :return: List of paths to extracted frames
        """
        frames_to_faces_map = defaultdict(list)
        for group_name, face_list in face_groups.items():
            for face in face_list:
                try:
                    # Parse the frame number from the face name
                    frame_number = int(face.name.split('_')[-1].split('.')[0])
                    frames_to_faces_map[frame_number].append(face)
                except (IndexError, ValueError):
                    self.logger.warning(f"Could not parse frame number from face name: {face.name}")
        
        return self._process_video_frames(video_path, frames_to_faces_map, output_directory)

    def _process_video_frames(self,
                               video_path: str,
                               frame_faces_mapping: Dict[int, List[Face]],
                               output_directory: str) -> tuple[List[str], str]:
        """
        Internal method to process video frames and extract relevant frames and cropped faces.
        :param video_path: Path to the video file
        :param frame_faces_mapping: Dictionary mapping frame numbers to lists of faces
        :param output_directory: Directory to save the extracted frames and cropped faces
        :return: List of paths to extracted frames and the path to the cropped faces directory
        """
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cropped_faces_directory = os.path.join(output_directory, "cropped_faces")
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(cropped_faces_directory, exist_ok=True)

        extracted_frame_paths = []
        video_capture = cv2.VideoCapture(video_path)

        try:
            if not video_capture.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            sorted_frame_numbers = sorted(set(frame_faces_mapping.keys()))
            target_frame_numbers = set(sorted_frame_numbers)
            current_frame_index = 0

            while video_capture.isOpened():

                ret = video_capture.grab() # get the next frame
                if not ret:
                    break  # End of video

                # Check if the current frame index is in the target frames
                if current_frame_index in target_frame_numbers:
                    # Read the current frame
                    _, frame = video_capture.retrieve()
                    associated_faces = frame_faces_mapping[current_frame_index]
                    output_frame_filename = associated_faces[0].name.split('_', 2)[-1]

                    # Save the full frame
                    full_frame_path = os.path.join(output_directory, output_frame_filename)
                    cv2.imwrite(full_frame_path, frame)
                    extracted_frame_paths.append(full_frame_path)

                    # Save cropped faces from this frame
                    for face in associated_faces:
                        try:
                            # Convert landmarks to numpy array if needed
                            landmarks = np.array(face.landmarks) if not isinstance(face.landmarks, np.ndarray) else face.landmarks
                            # Crop face using landmarks
                            cropped_face_image = norm_crop(frame, landmarks, 112, "arcface")
                            
                            # Save the cropped face
                            cropped_face_path = os.path.join(cropped_faces_directory, f"{face.name}")
                            cv2.imwrite(cropped_face_path, cropped_face_image)
                        except Exception as e:
                            self.logger.error(f"Error cropping face {face.name}: {e}")

                    # Remove the processed frame number from the target frames
                    target_frame_numbers.remove(current_frame_index)

                    # Exit the loop if all target frames are processed
                    if not target_frame_numbers:
                        break

                # Increment the frame counter
                current_frame_index += 1

            # Log completion
            self.logger.info("Frame extraction and cropping completed successfully.")

            # Warn if any frames were missed
            if target_frame_numbers:
                self.logger.warning(f"Could not find or process frames: {target_frame_numbers}")

            return extracted_frame_paths, cropped_faces_directory

        finally:
            video_capture.release()
