import os
import cv2
class MediaLoader:
    min_img_dim=112
    ALLOWED_IMG_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
    }
    ALLOWED_VID_EXTENSIONS = {
        ".mp4",
    }
    @staticmethod
    def allowed_img_file(filename: str):
        extension = os.path.splitext(filename)[1]
        return extension.lower() in MediaLoader.ALLOWED_IMG_EXTENSIONS

    @staticmethod
    def allowed_video_file(filename: str):
        extension = os.path.splitext(filename)[1]
        return extension.lower() in MediaLoader.ALLOWED_VID_EXTENSIONS

    @staticmethod
    def load_image( path: str):
        if(not MediaLoader.allowed_img_file(path)):
            return None;
        
        img = cv2.imread(path)
        height, width = img.shape[:2]

        if min(height, width) < MediaLoader.min_img_dim:
            # Calculate the scaling factor
            scale = MediaLoader.min_img_dim / min(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize the image
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return img       
    
    