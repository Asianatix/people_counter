import copy
import warnings

import numpy as np
import torch
from torchvision.ops.boxes import box_iou, nms

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

setup_logger()


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Detectron2:
    def __init__(self, cfg_path, weights_path):
        self.cfg = get_cfg()

        if cfg_path is None:
            print("Getting default config path for detectron")
            self.cfg.merge_from_file(
                "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        else:
            self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65  # Set min conf threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # NMS Threshold

        if weights_path is None:
            print("Getting default weights path for detectron ")
            self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        else:
            self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(self.cfg)
        self.batch_ensemble_thresh = 0.3

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

        return (
            np.array(bbox_xcycwh, dtype=np.float64),
            np.array(cls_conf),
            np.array(cls_ids),
        )

    def detect_batch(self, imgs, apply_batch_ensemble=False):
        batch_outputs = self.predictor.predict_batch_images(imgs)

        predictions = []
        for outputs in batch_outputs:
            boxes = outputs["instances"].pred_boxes.tensor
            classes = outputs["instances"].pred_classes
            scores = outputs["instances"].scores
            foreground_boxes = torch.where(
                classes >= 0,
                torch.ones(classes.size()).bool().cuda(),
                torch.zeros(classes.size()).bool().cuda(),
            )
            boxes = boxes[foreground_boxes]
            classes = classes[foreground_boxes]
            scores = scores[foreground_boxes]

            # predictions.append([f_boxes,scores[foreground_boxes], classes[foreground_boxes]])
            predictions.append([boxes, scores, classes])
        if apply_batch_ensemble:
            threshold = 0.3
            # all_box_status = [np.ones(len(f_p[0]), dtype=bool) for f_p in predictions]
            num_imgs = len(predictions)
            if not num_imgs < 2:
                final_predictions = predictions[0]
                for idx in range(1, num_imgs):
                    box1, sc1, conf1 = final_predictions
                    box2, sc2, conf2 = predictions[idx]

                    b_ious = box_iou(box1, box2)
                    mask = torch.where(
                        b_ious >= threshold,
                        torch.ones(b_ious.size()).bool().cuda(),
                        torch.zeros(b_ious.size()).bool().cuda(),
                    )

                    b1_mask = mask.any(axis=1)
                    b2_mask = mask.any(axis=0)
                    b1_boxs = box1[b1_mask]
                    b2_boxs = box2[b2_mask]
                    final_bbox = torch.cat([b1_boxs, b2_boxs], axis=0)

                    b1_sc = sc1[b1_mask]
                    b2_sc = sc2[b2_mask]
                    final_scores = torch.cat([b1_sc, b2_sc], axis=0)

                    b1_conf = conf1[b1_mask]
                    b2_conf = conf2[b2_mask]
                    final_conf = torch.cat([b1_conf, b2_conf], axis=0)

                    final_predictions = [final_bbox, final_scores, final_conf]
                boxs, scores, confs = final_predictions
                b_mask = nms(boxs, scores, iou_threshold=0.5)
                boxs = boxs[b_mask]
                scores = scores[b_mask]
                confs = confs[b_mask]
                predictions = [[boxs, scores, confs]]
        final_preds = []
        for frame_predictions in predictions:
            # From xyxy to xcycwh
            f_boxes, scores, confs = frame_predictions
            f_boxes[:, 0] = (f_boxes[:, 0] + f_boxes[:, 2]) / 2
            f_boxes[:, 1] = (f_boxes[:, 1] + f_boxes[:, 3]) / 2
            f_boxes[:, 2] = (f_boxes[:, 2] - f_boxes[:, 0]) * 2
            f_boxes[:, 3] = (f_boxes[:, 3] - f_boxes[:, 1]) * 2
            f_boxes = f_boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            confs = confs.cpu().numpy()
            final_preds.append([f_boxes, scores, confs])
        return final_preds


if __name__ == "__main__":
    import cv2
    import time
    import os
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("--vpath", default="../sample_videos/demo_2_40s.mp4")
    p.add_argument("--imp", default=None)
    p.add_argument(
        "-wp",
        "--weight_path",
        default="/nfs/gpu14_datasets/surveillance_weights/visdrone_t1/model_0111599.pth",
    )
    p.add_argument(
        "-cp",
        "--cfg_path",
        default="/nfs/gpu14_datasets/surveillance_weights/visdrone_t1/test.yaml",
    )
    args = p.parse_args()
    v_path = args.vpath

    w_p = args.weight_path
    cfg_p = args.cfg_path
    d = Detectron2(cfg_p, w_p)

    save_folder = "/mnt/nfshome1/FRACTAL/vikash.challa/BMC/iff/people_counter/frames"
    # save_folder = os.path.join(s_r, os.path.basename(v_path).replace(".mp4", "_") )
    os.system("mkdir -p {}".format(save_folder))

    print(v_path)
    cap = cv2.VideoCapture(v_path)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    try:
        im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20
    except Exception as e:
        im_width = 1280
        im_height = 786
        fps = 20
    video_output = cv2.VideoWriter(
        "/data/temp/sample.mp4", fourcc, fps, (im_width * 2, im_height)
    )
    assert cap.isOpened()
    frame_count = 100000

    def get_3_frames():
        ret_frames = []
        for i in range(3):
            f, im = cap.read()
            if f:
                ret_frames.append(im)
        return ret_frames

    from util import bbox_cxywh_xywh, draw_bboxes_xywh

    count = 0
    while True:
        ims = get_3_frames()
        frame_count += len(ims)
        if len(ims) < 1:
            break
        b_outs_e = d.detect_batch(ims, apply_batch_ensemble=True)
        b_outs = d.detect_batch(ims, apply_batch_ensemble=False)

        s_ims = ims
        e_ims = [copy.deepcopy(i) for i in ims]

        for idx, each_im_ouputs in enumerate(b_outs):
            bbox_xcycwh, cls_conf, cls_ids = each_im_ouputs
            persons_count = 0

            bbox_xywh = []
            if bbox_xcycwh is not None and bbox_xcycwh != []:
                mask = cls_ids == 0
                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2
                cls_conf = cls_conf[mask]
                persons_count = len(bbox_xcycwh)
                bbox_xywh = bbox_cxywh_xywh(bbox_xcycwh)
                if persons_count > 0:
                    im = s_ims[idx]
                    im = draw_bboxes_xywh(im, [bbox_xywh], None)
                    s_ims[idx] = im

        eam_im_ouput = b_outs_e[0]
        bbox_xcycwh, cls_conf, cls_ids = eam_im_ouput
        persons_count = 0

        bbox_xywh = []
        if bbox_xcycwh is not None and bbox_xcycwh != []:
            mask = cls_ids == 0
            bbox_xcycwh = bbox_xcycwh[mask]
            bbox_xcycwh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            persons_count = len(bbox_xcycwh)
            bbox_xywh = bbox_cxywh_xywh(bbox_xcycwh)
            if persons_count > 0:
                im = e_ims[-1]
                im = draw_bboxes_xywh(im, [bbox_xywh], None)
                e_ims[-1] = im

        for i in range(len(e_ims)):
            im = np.concatenate([e_ims[-1], s_ims[i]], axis=1)
            video_output.write(im)
        print(count)
        count += 3
    video_output.release()
