from ..base_detector_model import BaseDetectorModel
from models.detector_name import DetectorName
import os
import numpy as np
from insightface.app.common import Face
import cv2
import tritonclient.http as httpclient
from services.processing.triton.triton_client_handler import TritonClientHandler
from tritonclient.http._infer_result import InferResult
from ..detector_utils import nms_2, anchors_plane, bbox_pred, landmark_pred, clip_boxes
import time

class TritonEranRetinaFaceDetector(BaseDetectorModel):
    def __init__(self, root=""):
        self.name = DetectorName.eran_retinaface

    def detection_preprocessing(self, imgs: cv2.Mat) -> tuple[cv2.Mat, float]:

        im_tensor = np.zeros((imgs.shape[0], 3, imgs.shape[1], imgs.shape[2]), dtype=np.float32)
        for i in range(3):
            im_tensor[:, i, :, :] = imgs[:,:, :, 2 - i]

        return im_tensor

    def detection_postprocessing(
        self,
        scores_list: np.ndarray,
        bboxes_list: np.ndarray,
        kpss_list: np.ndarray,
        original_shape,
        det_scale,
    ) -> np.ndarray:
        dets = []
        kpss_all = []
        for batch_idx in range(len(scores_list)): 
            if len(scores_list[batch_idx])==0:
                dets.append(np.empty((0,5)))
                kpss_all.append(np.empty((0,5,2)))
                continue;
            scores = np.vstack(scores_list[batch_idx])
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            bboxes = np.vstack(bboxes_list[batch_idx]) / det_scale
            bboxes = bboxes[order, :]
            scores = scores[order]
            kpss = kpss_list[batch_idx] / det_scale
            kpss = kpss[order].astype(np.float32, copy=False)

            pre_det = np.hstack((bboxes[:, 0:4], scores)).astype(np.float32, copy=False)
            keep = nms_2(pre_det)
            det = np.hstack((pre_det, bboxes[:, 4:]))

            det = pre_det[keep, :]
            kpss = kpss[keep]
            dets.append(det)
            kpss_all.append(kpss)
        return dets, kpss_all
    

    def extract_from_response(
        self,response: InferResult, input_shape,threshold
    ) -> tuple[list, list, list]:
        _anchors_fpn = {
            32: np.array(
                [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]]
            ),
            16: np.array([[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]]),
            8: np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]]),
        }
        _feat_stride_fpn = [32, 16, 8]
        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        _num_anchors = 2
        landmark_std = 1.0
        N = input_shape[0]  # Batch size
        net_outs = []
        scores_list = [np.empty((0,1)) for _ in range(N)]
        bboxes_list = [ np.empty((0, 4)) for _ in range(N)]
        kpss_list = [ np.empty((0, 5, 2)) for _ in range(N)]
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
        im_info = [input_shape[2], input_shape[3]]
        for x in outputs:
            net_outs.append(response.as_numpy(x))
      
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
            scores = scores.transpose((0, 2, 3, 1)).reshape((N, -1, 1))  # Shape: (N, H * W * A, 1)

            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))  # Shape: (N, H, W, 4 * A)
            bbox_pred_len = bbox_deltas.shape[3] // A
            bbox_deltas = bbox_deltas.reshape((N, -1, bbox_pred_len))  # Shape: (N, H * W * A, 4)
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
            proposals = np.array([clip_boxes(proposal, im_info[:2]) for i, proposal in enumerate(proposals)])
            
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
                kpss_list[i]=np.concatenate([kpss_list[i],landmarks[i][pos_inds[i]]], axis=0)
                scores_list[i]=np.concatenate([scores_list[i],scores[i][pos_inds[i]]], axis=0)
                bboxes_list[i]=np.concatenate([bboxes_list[i],proposals[i][pos_inds[i]]], axis=0)
                
        return scores_list,bboxes_list,kpss_list

    def extract_faces(self, imgs:cv2.Mat,threshold=0.8) -> list[dict]:
        is_list=False
        if isinstance(imgs,list):
            is_list=True
            imgs=np.stack(imgs)
        else:
            imgs=np.expand_dims(imgs, axis=0)
        blob = self.detection_preprocessing(imgs)
        # blob=np.zeros((1,3,640,640),dtype=np.float32)
        # detection_response=session.run(outputs,{"data":input} )

        detection_input = httpclient.InferInput("data", blob.shape, datatype="FP32")
        detection_input.set_data_from_numpy(blob, binary_data=True)
        # start = time.time()
        detection_response=TritonClientHandler.infer(model_name="eran_detector",inputs=[detection_input])
        # end = time.time()
        # print(f"elapsed: {(end-start)*1000}ms")
        scores_list, bboxes_list, kpss_list = self.extract_from_response(
            detection_response, blob.shape,threshold=threshold
        )
        det, kpss = self.detection_postprocessing(
            scores_list, bboxes_list, kpss_list, blob.shape, 1.0
        )
        all_faces=[]
        for i in range(len(det)):#for each image
            faces=[Face(bbox=bbox[0:4],kps=kps,det_score=bbox[4]) for bbox,kps in zip(det[i],kpss[i])]
            if not is_list:
                return faces;
            all_faces.append(faces);
        return all_faces;