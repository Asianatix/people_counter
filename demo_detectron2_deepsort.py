import PIL
PIL.PILLOW_VERSION = PIL.__version__
import argparse
import os
import time
from distutils.util import strtobool
import copy

import pandas as pd

import cv2

from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))
        
        self.output_csv = args.csv_save_path

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        df = pd.DataFrame(columns=["time", "count"])
        df_list = []
        count = 0
        while self.vdo.grab():
            start = time.time()
            _, im = self.vdo.retrieve()
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if count % 3 == 0:
                bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)
                bbox_xcycwh_cpy = copy.deepcopy(bbox_xcycwh)
                cls_conf_cpy = copy.deepcopy(cls_conf)
                cls_ids_cpy = copy.deepcopy(cls_ids)
            else:
                bbox_xcycwh, cls_conf, cls_ids = bbox_xcycwh_cpy, cls_conf_cpy, cls_ids_cpy 

            if bbox_xcycwh is not None and bbox_xcycwh != []:
                # select class person
                mask = cls_ids == 0

                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2

                cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                df_list.append({"time": count/25, "count": len(outputs)})
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    im = draw_bboxes(im, bbox_xyxy, identities)

            end = time.time()
            print("count: {}, time: {}s, fps: {}".format(count,end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(im)
                new_df = pd.concat([df, pd.DataFrame(df_list)])
                new_df.to_csv(self.output_csv, index=False)
            # exit(0)
            count += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=False)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo2.mp4")
    parser.add_argument("--csv_save_path", default="demo.csv")
    parser.add_argument("--use_cuda", type=str, default="False")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
