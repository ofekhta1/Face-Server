from ..base_detector_model import BaseDetectorModel
from ..detector_utils import *
import numpy as np
import cv2
import tritonclient.http as httpclient
from tritonclient.http._infer_result import  InferResult
from Shared.services.util import filter_faces

from Shared.services.processing.triton.triton_client_handler import TritonClientHandler
from Shared.models.detector_name import DetectorName
from Shared.models.face import Face
import time

class TritonRetinaFace10GF(BaseDetectorModel):
    def __init__(self,root=""):
        self.name=DetectorName.retinaface_buffalo


    def detection_preprocessing(self,img: cv2.Mat,input_size =(640,640),input_std=128,input_mean=127.5) -> tuple[cv2.Mat,float]:
        if input_size is not None:
            im_ratio = float(img.shape[0]) / img.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio>model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)
            det_scale = float(new_height) / img.shape[0]
            resized_img = cv2.resize(img, (new_width, new_height))
            det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
            det_img[:new_height, :new_width, :] = resized_img
        else:
            det_img=img
            det_scale=1
        blob = cv2.dnn.blobFromImage(det_img, 1.0/input_std, input_size, (input_mean, input_mean, input_mean), swapRB=True)

        return blob,det_scale

    def detection_postprocessing(self,scores_list: np.ndarray, bboxes_list: np.ndarray, kpss_list: np.ndarray,original_shape,det_scale,max_num=0,
                             metric="default") -> np.ndarray:
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = original_shape[0] // 2, original_shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss
    def extract_from_response(self,response:InferResult,input_shape)->tuple[list,list,list]:
        fmc=3
        _feat_stride_fpn=[8,16,32]
        _num_anchors=2
        center_cache = {}
        net_outs=[];
        scores_list=[]
        bboxes_list=[]
        kpss_list=[]
        threshold=0.75
        #448-bbox6
        outputs=["448","471","494","451","474","497","454","477","500"]

        for x in outputs:
            net_outs.append(response.as_numpy(x))

        input_height =input_shape[2]
        input_width = input_shape[3]
    
        for idx, stride in enumerate(_feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)

            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if _num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*_num_anchors, axis=1).reshape( (-1,2) )
                if len(center_cache)<100:
                    center_cache[key] = anchor_centers
            pos_inds = np.where(scores>=threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss = distance2kps(anchor_centers, kps_preds)
            #kpss = kps_preds
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list   

    def _detect_with_size(self,img,input_size):
        blob,det_scale = self.detection_preprocessing(img, input_size=input_size) 
        detection_input = httpclient.InferInput("data", blob.shape, datatype="FP32")
        detection_input.set_data_from_numpy(blob, binary_data=True)
        detection_response=TritonClientHandler.infer(model_name=self.name,inputs=[detection_input])
        scores_list,bboxes_list,kpss_list= self.extract_from_response(detection_response,blob.shape)
        det,kpss=self.detection_postprocessing(scores_list,bboxes_list,kpss_list,img.shape,det_scale)
        return det,kpss

    def extract_faces(self,img)->list[dict]:
        start=time.time()
        far_results=self._detect_with_size(img,(1024,1024))
        close_results=self._detect_with_size(img,(320,320))
        close_faces=[Face(bbox=bbox[0:4],kps=kps,det_score=bbox[4]) for bbox,kps in zip(close_results[0],close_results[1])]
        far_faces=[Face(bbox=bbox[0:4],kps=kps,det_score=bbox[4]) for bbox,kps in zip(far_results[0],far_results[1])]
        end=time.time()
        print(f"took {(end-start)*1000}ms to detect faces")
        return filter_faces(close_faces,far_faces);

