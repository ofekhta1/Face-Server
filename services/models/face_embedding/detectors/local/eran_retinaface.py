import sys
import os
import traceback
from .retinaface50.retinaface import RetinaFace 
from ..base_detector_model import BaseDetectorModel
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from insightface.app.common import Face
import cv2
import numpy as np
from models.detector_name import DetectorName
import onnxruntime
from services.models.face_embedding.detectors.detector_utils import *


outputs = [
    "face_rpn_cls_prob_reshape_stride32",
    "face_rpn_bbox_pred_stride32",
    "face_rpn_landmark_pred_stride32",
    "face_rpn_cls_prob_reshape_stride16",
    "face_rpn_bbox_pred_stride16",
    "face_rpn_landmark_pred_stride16",
    "face_rpn_cls_prob_reshape_stride8",
    "face_rpn_bbox_pred_stride8",
    "face_rpn_landmark_pred_stride8",
]
_anchors_fpn = {
    32: np.array([[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]]),
    16: np.array([[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]]),
    8: np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]]),
}
_feat_stride_fpn = [32, 16, 8]
bbox_stds = [1.0, 1.0, 1.0, 1.0]
_num_anchors = 2
landmark_std = 1.0

rec_input_size = (112, 112)
rec_input_std = 127.5
rec_input_mean = 127.5


def preprocess_eran_detector(imgs):
    max_shape = (1080, 1920)
    det_scales = [1] * imgs.shape[0]
    if imgs.shape[1] > max_shape[0] or imgs.shape[2] > max_shape[1]:
        im_tensor = np.zeros(
            (imgs.shape[0], 3, max_shape[0], max_shape[1]), dtype=np.float32
        )
        for i in range(imgs.shape[0]):
            img: cv2.Mat = imgs[i]
            height, width = img.shape[:2]
            scaling_factor = min(max_shape[1] / width, max_shape[0] / height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            det_img = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

            det_scales[i] = scaling_factor

            for j in range(3):
                im_tensor[i, j, :new_height, :new_width] = det_img[:, :, 2 - j]
    else:
        im_tensor = np.zeros(
            (imgs.shape[0], 3, imgs.shape[1], imgs.shape[2]), dtype=np.float32
        )
        for i in range(3):
            im_tensor[:, i, :, :] = imgs[:, :, :, 2 - i]
    return im_tensor, det_scales


def extract_from_eran_detector_response(net_outs:list, N=1, input_shape=(1024, 1024),threshold=0.8):

    scores_list = [np.empty((0, 1)) for _ in range(N)]
    bboxes_list = [np.empty((0, 4)) for _ in range(N)]
    kpss_list = [np.empty((0, 5, 2)) for _ in range(N)]

    for idx, stride in enumerate(_feat_stride_fpn):
        scores = net_outs[idx * 3]
        scores = scores[:, _num_anchors:, :, :]  # Shape: (N, A, H, W)

        bbox_deltas = net_outs[(idx * 3) + 1]  # Shape: (N, 4 * A, H, W)
        height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]
        A = _num_anchors
        K = height * width
        anchors_fpn = _anchors_fpn[stride]
        anchors = anchors_plane(height, width, stride, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))

        # Adjust scores and bbox_deltas for batched input
        scores = scores.transpose((0, 2, 3, 1)).reshape(
            (N, -1, 1)
        )  # Shape: (N, H * W * A, 1)

        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))  # Shape: (N, H, W, 4 * A)
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape(
            (N, -1, bbox_pred_len)
        )  # Shape: (N, H * W * A, 4)
        bbox_deltas[:, :, 0::4] *= bbox_stds[0]
        bbox_deltas[:, :, 1::4] *= bbox_stds[1]
        bbox_deltas[:, :, 2::4] *= bbox_stds[2]
        bbox_deltas[:, :, 3::4] *= bbox_stds[3]

        # Apply bbox_pred for each image in the batch
        proposals = []
        for i in range(N):
            proposals.append(bbox_pred(anchors, bbox_deltas[i]))
        proposals = np.stack(proposals, axis=0)  # Shape: (N, H * W * A, 4)

        # Clip proposals to image dimensions
        proposals = np.array(
            [clip_boxes(proposal, input_shape) for i, proposal in enumerate(proposals)]
        )

        # Filter out low-confidence proposals
        scores_ravel = scores.reshape(N, -1)
        pos_inds = [np.where(scores_ravel[i] >= threshold)[0] for i in range(N)]

        # Process landmarks
        landmark_deltas = net_outs[idx * 3 + 2]  # Shape: (N, 10 * A, H, W)
        landmark_pred_len = landmark_deltas.shape[1] // A
        landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape(
            (N, -1, 5, landmark_pred_len // 5)
        )
        landmark_deltas *= landmark_std

        # Apply landmark_pred for each image in the batch
        landmarks = []
        for i in range(N):
            landmarks.append(landmark_pred(anchors, landmark_deltas[i]))
        landmarks = np.stack(landmarks, axis=0)  # Shape: (N, H * W * A, 10)
        for i in range(N):
            # Filter landmarks based on positive indices
            kpss_list[i] = np.concatenate(
                [kpss_list[i], landmarks[i][pos_inds[i]]], axis=0
            )
            scores_list[i] = np.concatenate(
                [scores_list[i], scores[i][pos_inds[i]]], axis=0
            )
            bboxes_list[i] = np.concatenate(
                [bboxes_list[i], proposals[i][pos_inds[i]]], axis=0
            )

    return scores_list, bboxes_list, kpss_list


arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def eran_detection_postprocessing(
    scores_list: np.ndarray,
    bboxes_list: np.ndarray,
    kpss_list: np.ndarray,
    det_scales: list[float],
) -> np.ndarray:
    dets = []
    kpss_all = []
    for batch_idx in range(len(scores_list)):
        if len(scores_list[batch_idx]) == 0:
            dets.append(np.empty((0, 5)))
            kpss_all.append(np.empty((0, 5, 2)))
            continue
        scores = np.vstack(scores_list[batch_idx])
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list[batch_idx]) / det_scales[batch_idx]
        bboxes = bboxes[order, :]
        scores = scores[order]
        kpss = kpss_list[batch_idx] / det_scales[batch_idx]
        kpss = kpss[order].astype(np.float32, copy=False)

        pre_det = np.hstack((bboxes[:, 0:4], scores)).astype(np.float32, copy=False)
        keep = nms_2(pre_det)
        det = np.hstack((pre_det, bboxes[:, 4:]))

        det = pre_det[keep, :]
        kpss = kpss[keep]
        dets.append(det)
        kpss_all.append(kpss)
    return dets, kpss_all


def post_process_eran_detector(imgs, net_outs, det_scales,threshold=0.8):

    scores_list, bbox_list, kps_list = extract_from_eran_detector_response(
        net_outs, input_shape=(imgs[0].shape[1], imgs[0].shape[2]),threshold=threshold
    )
    det, kpss = eran_detection_postprocessing(
        scores_list, bbox_list, kps_list, det_scales
    )
    
    return  det, kpss

def preprocess_eran_detector(imgs):
    max_shape = (1080, 1920)
    det_scales = [1] * imgs.shape[0]
    if imgs.shape[1] > max_shape[0] or imgs.shape[2] > max_shape[1]:
        im_tensor = np.zeros(
            (imgs.shape[0], 3, max_shape[0], max_shape[1]), dtype=np.float32
        )
        for i in range(imgs.shape[0]):
            img: cv2.Mat = imgs[i]
            height, width = img.shape[:2]
            scaling_factor = min(max_shape[1] / width, max_shape[0] / height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            det_img = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

            det_scales[i] = scaling_factor

            for j in range(3):
                im_tensor[i, j, :new_height, :new_width] = det_img[:, :, 2 - j]
    else:
        im_tensor = np.zeros(
            (imgs.shape[0], 3, imgs.shape[1], imgs.shape[2]), dtype=np.float32
        )
        for i in range(3):
            im_tensor[:, i, :, :] = imgs[:, :, :, 2 - i]
    return im_tensor, det_scales

class LocalEranRetinaFaceDetector(BaseDetectorModel):
    def __init__(self,root=""):
        self.name=DetectorName.eran_retinaface
        self.model_name = os.path.join(root,"OnnxModels","Detectors","eran_retinaface.onnx") # Use the face recognition model
        self.session=onnxruntime.InferenceSession(self.model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    def __extract_faces_internal(self,img:cv2.Mat,resize=False,input_size=(640,640)):
        try:
            im_tensor,det_scales=preprocess_eran_detector(np.array([img]))
            net_outs = self.session.run(outputs,{"data":im_tensor});

            dets,kpss=post_process_eran_detector(im_tensor,net_outs,det_scales)
            faces=[Face(bbox=bbox[0:4],kps=kps,det_score=bbox[4]) for bbox,kps in zip(dets[0],kpss[0])]#currently only do it for one image so [0]

            return faces
        except Exception as e:
            print("Error during face extraction:", e)
            return None;

    def extract_faces(self,img:np.ndarray):
        faces=self.__extract_faces_internal(img)
        # far_faces=self.__extract_faces_internal(img,self.input_size_zoomed)
        # faces=far_faces.copy(); 
        # for j in range(len(close_faces)):
        #     duplicate=False;
        #     for far_face in far_faces:
        #         if(are_bboxes_similar(close_faces[j]['bbox'],far_face['bbox'],20)):
        #             duplicate=True;
        #     if(not duplicate):
        #         faces.append(close_faces[j])
 
        return faces