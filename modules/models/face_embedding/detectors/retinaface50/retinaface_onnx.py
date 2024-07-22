from __future__ import print_function
import sys
import numpy as np
import cv2
sys.path.append("retinaface50/")
from .rcnn.processing.bbox_transform import clip_boxes
from .rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from .rcnn.processing.nms import gpu_nms_wrapper
import onnxruntime
from onnxruntime import ExecutionMode
import time



class RetinaFaceOnnx:
    def __init__(self,model_path):
        self.model_path=model_path
        session_options = onnxruntime.SessionOptions()
        session_options.enable_profiling = True
        session_options.intra_op_num_threads = 10
        session_options.execution_mode  = ExecutionMode.ORT_PARALLEL
        self.session = onnxruntime.InferenceSession(model_path, session_options,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],verbose=True)
        _ratio=(1.,)

        self.bbox_stds = [1.0, 1.0, 1.0, 1.0]
        self.landmark_std = 1.0
        self._feat_stride_fpn=[32, 16, 8]
        self.fpn_keys=[]
        self.nms = gpu_nms_wrapper(0.4,1)
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        self.anchor_cfg = {
                '32': {
                    'SCALES': (32, 16),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '16': {
                    'SCALES': (8, 4),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '8': {
                    'SCALES': (2, 1),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
            }

        self._anchors_fpn = dict(
            zip(
                self.fpn_keys,
                generate_anchors_fpn(dense_anchor=False,
                                     cfg=self.anchor_cfg)))
        
      
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v

        self._num_anchors = dict(
            zip(self.fpn_keys,
                [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
    def stop_profiling(self):
        self.session.end_profiling();
    def detect(self, img, threshold=0.8):
        proposals_list = []
        scores_list = []
        landmarks_list = []

        imgs = [img]
        if isinstance(img, list):
            imgs = img
        for img in imgs:
            im = img.astype(np.float32)
            
            im_info = [im.shape[0], im.shape[1]]
            im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]),dtype=np.float32)
            for i in range(3):
                im_tensor[0, i, :, :] = im[:, :, 2 - i] 
            start=time.time()
            onnxruntime.set_default_logger_severity(0)
            runOptions=onnxruntime.RunOptions();
            runOptions.log_severity_level=1
            outputs = self.session.run(["face_rpn_cls_prob_reshape_stride32","face_rpn_bbox_pred_stride32","face_rpn_landmark_pred_stride32",
                                        "face_rpn_cls_prob_reshape_stride16","face_rpn_bbox_pred_stride16","face_rpn_landmark_pred_stride16",
                                        "face_rpn_cls_prob_reshape_stride8","face_rpn_bbox_pred_stride8","face_rpn_landmark_pred_stride8"],
                                        {"data": im_tensor},run_options=runOptions)
            end=time.time()
            print(end-start)
            sym_idx = 0

            for _idx, s in enumerate(self._feat_stride_fpn):
                #if len(scales)>1 and s==32 and im_scale==scales[-1]:
                #  continue
                _key = 'stride%s' % s
                stride = int(s)
                is_cascade = False
        
            
                scores = outputs[sym_idx]
        
                scores = scores[:, self._num_anchors['stride%s' %
                                                        s]:, :, :]

                bbox_deltas = outputs[sym_idx + 1]

        
                height, width = bbox_deltas.shape[
                    2], bbox_deltas.shape[3]

                A = self._num_anchors['stride%s' % s]
                K = height * width
                anchors_fpn = self._anchors_fpn['stride%s' % s]
                anchors = anchors_plane(height, width, stride,
                                        anchors_fpn)
                anchors = anchors.reshape((K * A, 4))
                
                scores = scores.transpose((0, 2, 3, 1)).reshape(
                    (-1, 1))

        
                bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                bbox_pred_len = bbox_deltas.shape[3] // A
                bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                bbox_deltas[:,
                            0::4] = bbox_deltas[:, 0::
                                                4] * self.bbox_stds[0]
                bbox_deltas[:,
                            1::4] = bbox_deltas[:, 1::
                                                4] * self.bbox_stds[1]
                bbox_deltas[:,
                            2::4] = bbox_deltas[:, 2::
                                                4] * self.bbox_stds[2]
                bbox_deltas[:,
                            3::4] = bbox_deltas[:, 3::
                                                4] * self.bbox_stds[3]
                proposals = self.bbox_pred(anchors, bbox_deltas)

                #print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
                
                proposals = clip_boxes(proposals, im_info[:2])

                if stride == 4 and self.decay4 < 1.0:
                    scores *= self.decay4

                scores_ravel = scores.ravel()
                
                order = np.where(scores_ravel >= threshold)[0]
                
                proposals = proposals[order, :]
                scores = scores[order]
                


                proposals_list.append(proposals)
                scores_list.append(scores)

                landmark_deltas = outputs[sym_idx + 2]
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose(
                    (0, 2, 3, 1)).reshape(
                        (-1, 5, landmark_pred_len // 5))
                landmark_deltas *= self.landmark_std
                landmarks = self.landmark_pred(
                    anchors, landmark_deltas)
                landmarks = landmarks[order, :]

                landmarks_list.append(landmarks)
                sym_idx += 3


        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = np.vstack(proposals_list)
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)


        proposals = proposals[order, :]
        scores = scores[order]
        
        
        pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32,
                                                                copy=False)
        keep = self.nms(pre_det)
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        landmarks = landmarks[keep]
   
        return det, landmarks


    @staticmethod
    def bbox_pred(boxes, box_deltas):
        """
      Transform the set of class-agnostic boxes into class-specific boxes
      by applying the predicted offsets (box_deltas)
      :param boxes: !important [N 4]
      :param box_deltas: [N, 4 * num_classes]
      :return: [N 4 * num_classes]
      """
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        # x1
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        # y1
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        # x2
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        # y2
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1] > 4:
            pred_boxes[:, 4:] = box_deltas[:, 4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
            pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
        return pred
