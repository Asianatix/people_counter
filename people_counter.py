import PIL
try:
    _ = PIL.PILLO_VERSION
except:
    PIL.PILLOW_VERSION = PIL.__version__
import argparse
import argparse
import os
import time
from distutils.util import strtobool
import copy
from datetime import timedelta

import pandas as pd
import numpy as np

import cv2

from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()
        self.count_thresh = args.count_thresh
        self.entry_freq = args.entry_freq

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
     #   assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
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
        thresh_column_name = "count>{}".format(self.count_thresh)
        # thresh_column_name = "countgt5"
        df = pd.DataFrame(columns=["time", "count", thresh_column_name])
        df_list = []
        count_list = []
        count = 0
        try:

            fps = int(self.vdo.get(cv2.CAP_PROP_FPS))
        except:
            print("RTSP stream ")
            fps = 1
        init_time = time.time()
        status, im = self.vdo.read()
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
                # outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                outputs = self.deepsort.update_new(bbox_xcycwh, cls_conf, im)
                if count%(self.entry_freq*fps)==0:
                    count_list.append(len(outputs))
                    tme = str(timedelta(seconds=count/fps))
                    ct = int(np.mean(count_list))
                    desc = ct>self.count_thresh
                    print ("&&&&", tme, ct, desc)
                    df_list.append({"time": tme, "count": ct, thresh_column_name: desc})
                    count_list = []
                else:
                    count_list.append(len(outputs))

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    im = draw_bboxes(im, bbox_xyxy, identities)
            else:
                if count%(self.entry_freq*fps)==0:
                    count_list.append(0)
                    tme = str(timedelta(seconds=count/fps))
                    ct = 0
                    desc = ct>self.count_thresh
                    print ("&&&&", tme, ct, desc)
                    df_list.append({"time": tme, "count": ct, thresh_column_name: desc})
                    count_list = []
                else:
                    count_list.append(0)
            cv2.imshow("preview", im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            end = time.time()
            avg_fps = (count+1)//(end - init_time)
            print("Procressing frame num : {}, time : {} avg-fps: {}".format(count,end - start, avg_fps))
            

            if self.args.save_path:
                self.output.write(im)
                new_df = pd.concat([df, pd.DataFrame(df_list)])
                new_df.to_excel(self.output_csv, index=False, encoding='utf-8-sig')
            # exit(0)
            count += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH",  default=0)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=False)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.mp4")
    parser.add_argument("--csv_save_path", default="demo.xlsx")
    parser.add_argument("--count_thresh", default=5)
    parser.add_argument("--entry_freq", help="frequency in seconds with which the data will enter in csv", default=1)
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--rtsp", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
