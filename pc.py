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
from util import draw_bboxes, get_bbox_xywh
import json 
from TCP.TCPClient import TCPClient

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        if self.args.tcp_ip_port is not None:
            self._set_tcp_client()
    
    def _set_tcp_client(self):
        ip, port = self.args.tcp_ip_port.strip().split(':')
        port = int(port)
        self.tcp_client = TCPClient(ip, port)
        self.tcp_client.LaunchConnection()

    def _set_video_writer(self, video_path):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_output = cv2.VideoWriter(video_path, fourcc, 20, (self.im_width, self.im_height))
        
    def __enter__(self):
     #   assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.video_path)
        self.vd_name = os.path.basename(self.args.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.args.save_video_to:
            self._set_video_writer("{}/0_{}_{}".format(self.args.save_video_to,self.args.save_video_freq, self.vd_name))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def _get_frame_dict(self):
        return dict(
            frame_id = "",
            people_count = 0,
            crowd_flag = False,
            bboxes_list = [] 
        )

    def detect(self):
        thresh_column_name = "Is crowd > {}".format(self.args.people_count_thresh)
        df = pd.DataFrame(columns=["Frame ID", "people count", thresh_column_name])
        df_list = []
        count_list = []
        try:
            fps = int(self.vdo.get(cv2.CAP_PROP_FPS))
        except:
            fps = None
            print("FPS not found in video meta data. Video path is not found or r rtsp stream  ")
            
        
        self.frame_count = 0
        self.processing_frame_count = 0
        
        model_avg_time = 0.0
        model_total_time = 0.0

        proc_avg_time = 0.0
        proc_total_time = 0.0
        init_time = time.time()
        bbox_cords_cpy = []
        while self.vdo.grab():
            frame_start_time = time.time()
            _, im = self.vdo.retrieve()
            j_dict = self._get_frame_dict()
            j_dict["frame_id"] = self.frame_count
            f_h, f_w = im.shape[:2]
            if self.frame_count % self.args.proc_freq == 0:
                self.processing_frame_count += 1
                model_init_time = time.time()
                bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)
                model_end_time = time.time()
                #print("Model time: {:.4f}".format(model_end_time - model_init_time))
                model_avg_time = (model_total_time + (model_end_time - model_init_time))/self.processing_frame_count
                bbox_xcycwh_cpy = copy.deepcopy(bbox_xcycwh)
                cls_conf_cpy = copy.deepcopy(cls_conf)
                cls_ids_cpy = copy.deepcopy(cls_ids)
                
            else: # Taking the last processed frame as its 
                bbox_xcycwh, cls_conf, cls_ids = bbox_xcycwh_cpy, cls_conf_cpy, cls_ids_cpy 

            #Some persons are found
            if bbox_xcycwh is not None and bbox_xcycwh != []:
                # select class person
                mask = cls_ids == 0
                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2
                cls_conf = cls_conf[mask]
                
                outputs = self.deepsort.update_new(bbox_xcycwh, cls_conf, im)
                if self.frame_count%(self.args.csv_save_freq)==0:
                    count_list.append(len(outputs))
                    ct = int(np.mean(count_list))
                    crowd_flag = ct > self.args.people_count_thresh
                    #print ("\t\tFrame: {} poeple_count: {} : crowd: {}".format(self.frame_count, ct, crowd_flag))
                    df_list.append({"frame_id": self.frame_count, "people count": ct, thresh_column_name: crowd_flag})
                    count_list = []
                else:
                    count_list.append(len(outputs))
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    bbox_xywh = get_bbox_xywh(bbox_xyxy, identities)
                    bbox_cords_cpy = copy.deepcopy(bbox_xywh)
                # If display is true display to cv window.
                if self.args.display or  self.args.save_video_to or (self.args.save_frames_to is not None):
                    if len(outputs) > 0:
                        im = draw_bboxes(im, bbox_xyxy, identities)
            # No people found 
            else:
                if self.frame_count%(self.args.csv_save_freq)==0:
                    count_list.append(0)
                    ct = 0
                    crowd_flag = ct > self.args.people_count_thresh
                   
                    df_list.append({"frame_id": self.frame_count, "people count": ct, "Is crowd > {}".format(self.args.people_count_thresh): crowd_flag})
                    count_list = []
                else:
                    count_list.append(0)
           
            if self.args.display:
                cv2.imshow("Live preview", im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
            self.frame_count += 1
            if self.args.save_video_to:
                self.video_output.write(im)
                
                if self.frame_count % self.args.save_video_freq == 0:
                    self.video_output.release()
                    print("Video saved")
                    self._set_video_writer("{}/{}_{}_{}".format(self.args.save_video_to, self.frame_count, self.frame_count + self.args.save_video_freq, self.vd_name))
            if self.args.save_frames_to is not None:
                frame_path = "{}/{}.jpg".format(self.args.save_frames_to, self.frame_count)
                print(frame_path)
                cv2.imwrite(frame_path, im)
                
            if self.args.save_csv_path:
                new_df = pd.concat([df, pd.DataFrame(df_list)])
                new_df.to_excel(self.args.save_csv_path, index=False, encoding='utf-8-sig')
            proc_total_time = proc_total_time + (time.time() - frame_start_time)
            
            if self.args.tcp_ip_port is not None:
                frame_bbox_flat = []
                for bbox in bbox_cords_cpy:
                    bbox = [bbox[0]/f_w, bbox[1]/f_h, bbox[2]/f_w, bbox[3]/f_h]
                    frame_bbox_flat += bbox
                try:
                    if not len(frame_bbox_flat) == 0:
                        print("Sending {}".format(frame_bbox_flat))
                    self.tcp_client.SendBoundingBoxes(frame_bbox_flat)
                except Exception as e:
                    print(e)
                    print("Unable to send data to TCP server")


            if not self.args.supress_verbose:
                
                model_avg_fps = self.frame_count // proc_total_time
                proc_avg_time = proc_total_time / self.frame_count
                frame_time = (time.time() - frame_start_time)
                nn_time = model_end_time - model_init_time
                actual_fps =   self.frame_count // (time.time() - init_time)
                print ("\t\tFrame:{} poeple_count:{} : crowd:{} Model_FPS:{} Actual_FPS:{} nn_time/avg:[{:.4f}/{:.4f}], frame_time/avg:[{:.4f}/{:.4f}]".format(
                    self.frame_count,
                    ct,
                    crowd_flag,
                    model_avg_fps,
                    actual_fps,
                    nn_time,
                    model_avg_time,
                    frame_time,
                    proc_avg_time))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",  default="sample_videos/demo.mp4", help="Path to video or path to rtsp stream")
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--proc_freq", type=int, default=3)
    parser.add_argument("--display",  action="store_true",help="To display on cv window")
    parser.add_argument("--save_video_to", type=str, default=None, help="Save path to video")
    parser.add_argument("--save_video_freq", type=int, default=100000, help="Save video at this number of frames")
    parser.add_argument("--save_csv_path", default="demo.xlsx")
    parser.add_argument("--people_count_thresh", default=5)
    parser.add_argument("--csv_save_freq", help="frequency in seconds with which the data will enter in csv", default=1)
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--supress_verbose", action="store_true", help="Supress print statements ")
    parser.add_argument("--tcp_ip_port", type=str, help="IP:PORT of tcp server", default=None)
    parser.add_argument("--save_frames_to",default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
