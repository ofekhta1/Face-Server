def transform_img_coords(
    det: list[float], 
    kps: list[list[float]], 
    scale: float, 
    paddings: tuple[float, float]
) -> tuple[list[float], list[tuple[float, float]]]:
    """
    Transform detection and keypoint coordinates from resized to original image coordinates.
    
    Args:
        det (list[float]): Detection coordinates [x1, y1, x2, y2, confidence]
        kps (list[list[float]]): Keypoints coordinates [(x1, y1), (x2, y2), ...]
        scale (float): Scaling factor used during resizing
        paddings (tuple[float, float]): Horizontal and vertical paddings applied
    
    Returns:
        tuple: Transformed detection coordinates and keypoints
    """
    # Convert landmarks back to original coordinates
    landmarks_original = [
        (
            (x_resized - paddings[0]) / scale, 
            (y_resized - paddings[1]) / scale
        ) 
        for x_resized, y_resized in kps
    ]
    
    # Transform detection box coordinates
    dets_original = [
        (det[0] - paddings[0]) / scale,  # x1
        (det[1] - paddings[1]) / scale,  # y1
        (det[2] - paddings[0]) / scale,  # x2
        (det[3] - paddings[1]) / scale,  # y2
        det[4]  # confidence score
    ]
    
    return dets_original, landmarks_original

def transform_imgs_coords(
    dets: list[list[float]], 
    kpss: list[list[list[float]]], 
    scale: float, 
    paddings: tuple[float, float]
) -> tuple[list[list[float]], list[list[tuple[float, float]]]]:
    """
    Transform multiple detections and keypoints from resized to original image coordinates.
    
    Args:
        dets (list[list[float]]): List of detection coordinates
        kpss (list[list[list[float]]]): List of keypoints for each detection
        scale (float): Scaling factor used during resizing
        paddings (tuple[float, float]): Horizontal and vertical paddings applied
    
    Returns:
        tuple: Transformed detections and keypoints
    """
    new_dets, new_kpss = zip(*[
        transform_img_coords(det, kps, scale, paddings) 
        for det, kps in zip(dets, kpss)
    ])
    
    return list(new_dets), list(new_kpss)