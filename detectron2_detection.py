from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class Detectron2:

    def __init__(self, cfg_path, weights_path):
        self.cfg = get_cfg()
        
        if cfg_path is None:
            print("Getting default config path for detectron")
            self.cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        else:
            self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        
        if weights_path is None:
            print("Getting default weights path for detectron ")
            self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        else:
            self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(self.cfg)

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax

    def detect(self, im):
        outputs = self.predictor(im)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        bbox_xcycwh, cls_conf, cls_ids = [], [], []

        for (box, _class, score) in zip(boxes, classes, scores):

            if _class >= 0:
                x0, y0, x1, y1 = box
                bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
                cls_conf.append(score)
                cls_ids.append(_class)

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids)

    def detect_batch(self, imgs, apply_batch_ensemble = False):
        batch_outputs = self.predictor.predict_batch_images(imgs)
        
        predictions = []
        for outputs in batch_outputs:
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            classes = outputs["instances"].pred_classes.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()

            bbox_xcycwh, cls_conf, cls_ids = [], [], []
            # for (box, _class, score) in zip(boxes, classes, scores):
            #     if _class >= 0:
            #         x0, y0, x1, y1 = box
            #         bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
            #         cls_conf.append(score)
            #         cls_ids.append(_class)
            # predictions.append([np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids)])
            
            
            foreground_boxes = np.where(classes >= 0, True, False)
            f_boxes = boxes[foreground_boxes]
            f_boxes[:, 0] = (f_boxes[:, 0] + f_boxes[:, 2])/2
            f_boxes[:, 1] = (f_boxes[:, 1] + f_boxes[:, 3])/2
            f_boxes[:, 2] = (f_boxes[:, 2] - f_boxes[:, 0])*2
            f_boxes[:, 0] = (f_boxes[:, 3] - f_boxes[:, 1])*2
            predictions.append([f_boxes,scores[foreground_boxes], classes[foreground_boxes]])
        if apply_batch_ensemble:
            pass
            
        return predictions
    

    
if __name__ == "__main__":
    import cv2
    import time
    d = Detectron2(None, None)
    from util import VideoCapture
    cap = VideoCapture("../sample_videos/demo_2_40s.mp4", buffer_size= 3)
    while True:
        f, ims = cap.read()
        b_outs = d.detect_batch(ims)
        
        
        
    
    
    
    