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
from util import draw_bboxes, get_bbox_xywh, agrregate_split_results, draw_bboxes_xywh
import json 
from TCP.TCPClient import TCPClient
import queue, threading, time
import math 

class VideoCapture:

  def __init__(self, video_path):
    self.cap = cv2.VideoCapture(video_path)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    self.fr_count = 0
    while True:
      ret, frame = self.cap.read()
      self.fr_count += 1
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        
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
        if self.args.buffer_frames:
            self.vdo = cv2.VideoCapture(self.args.video_path) 
            assert self.vdo.isOpened()
        else:
            self.vdo = VideoCapture(self.args.video_path)
            assert self.vdo.cap.isOpened()
        self.vd_name = os.path.basename(self.args.video_path)
        if self.args.save_video_to:
            self._set_video_writer("{}/0_{}_{}".format(self.args.save_video_to,self.args.save_video_freq, self.vd_name))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
            
    def detect_im(self, im):
        bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)
        
        persons_count  = 0
        # Some objects are found 
        bbox_xywh = []
        if bbox_xcycwh is not None and bbox_xcycwh != []:    
            mask = cls_ids == 0
            bbox_xcycwh = bbox_xcycwh[mask]
            bbox_xcycwh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            outputs = self.deepsort.update_new(bbox_xcycwh, cls_conf, im)
            persons_count = len(outputs)
            if persons_count > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                bbox_xywh = get_bbox_xywh(bbox_xyxy, identities)
        return bbox_xywh
    
    def split_detect(self, im):
        h, w = im.shape[:2]
        h_h = math.floor(h/2)
        h_w = math.floor(w/2)
        lt = im[:h_w, :h_h,]
        rt = im[h_w:, :h_h,]
        lb = im[:h_w, h_h:]
        rb = im[h_w:, h_h:]
        
        lt_out = self.detect_im(lt)
        rt_out = self.detect_im(rt)
        lb_out = self.detect_im(lb)
        rb_out = self.detect_im(rb)
        bbox_xywh = agrregate_split_results([lt_out, rt_out, lb_out, rb_out], h_w, h_h)
        return bbox_xywh
                
    def detect_video(self):    
        self.frame_count = 0
        self.processing_frame_count = 0
        persons_count = 0
        
        model_avg_time = 0.0
        model_total_time = 0.0

        proc_avg_time = 0.0
        proc_total_time = 0.0
        init_time = time.time()
        
        while True:
            frame_start_time = time.time()
            self.frame_count += 1
            
            if self.args.buffer_frames:
                _, im = self.vdo.read()
            else:
                im = self.vdo.read()
                
            f_h, f_w = im.shape[:2]
            
            if (self.frame_count-1) % self.args.proc_freq == 0:
                self.processing_frame_count += 1
                model_init_time = time.time()
                if not self.args.split_detector:                
                    bbox_xywh = self.detect_im(im)
                else:
                    bbox_xywh = self.split_detect(im)
                model_end_time = time.time()
                persons_count = len(bbox_xywh)
            if persons_count > 0 and (self.args.display or  self.args.save_video_to or (self.args.save_frames_to is not None)):    
                im = draw_bboxes_xywh(im, bbox_xywh, None)
                #im = draw_bboxes(im, bbox_xyxy, identities)
                            
            if self.args.display:
                cv2.imshow("Live preview", im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if self.args.save_video_to:
                self.video_output.write(im)
                if self.frame_count % self.args.save_video_freq == 0:
                    self.video_output.release()
                    print("Video saved")
                    self._set_video_writer("{}/{}_{}_{}".format(self.args.save_video_to, self.frame_count, self.frame_count + self.args.save_video_freq, self.vd_name))
                    
            if self.args.save_frames_to is not None and persons_count > 0:
                frame_path = "{}/{}.jpg".format(self.args.save_frames_to, self.frame_count)
                print(frame_path)
                cv2.imwrite(frame_path, im)
                
            proc_total_time = proc_total_time + (time.time() - frame_start_time)
            
            if self.args.tcp_ip_port is not None:
                
                frame_bbox_flat = []
                if persons_count > 0:
                    for bbox in bbox_xywh:
                        print(bbox)
                        bbox_ = [bbox[0]/f_w, bbox[1]/f_h, bbox[2]/f_w, bbox[3]/f_h]
                        frame_bbox_flat += bbox_
                try:
                    if len(frame_bbox_flat) >  0:
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
                if not self.args.buffer_frames:
                    video_fps = self.vdo.fr_count // (time.time() - init_time)
                    cap_f_count = self.vdo.fr_count
                else:
                    cap_f_count = self.frame_count
                    video_fps = self.frame_count // (time.time() - init_time)
                print ("cap_frame:{} p_Frame:{} p_count:{} :  M_FPS:{} cap_FPS:{:.4f} Process_FPS:{} nn_time/avg:[{:.4f}/{:.4f}], frame_time/avg:[{:.4f}/{:.4f}]".format(
                    cap_f_count, 
                    self.frame_count,
                    persons_count,
                    model_avg_fps,
                    video_fps, 
                    actual_fps,
                    nn_time,
                    model_avg_time,
                    frame_time,
                    proc_avg_time))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",  default="sample_videos/demo.mp4", help="Path to video or path to rtsp stream")
    parser.add_argument("--buffer_frames", action="store_true", help="If set true, buffer frames to process one after the other.Setting this true doesn't run in real-time.")
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--proc_freq", type=int, default=1, help="useful when running on old videos/ non-real time videos")
    parser.add_argument("--display",  action="store_true",help="To display on cv window")
    parser.add_argument("--save_video_to", type=str, default=None, help="Save path to video")
    parser.add_argument("--save_video_freq", type=int, default=100000, help="Save video at this number of frames")
    parser.add_argument("--people_count_thresh", default=5)
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--supress_verbose", action="store_true", help="Supress print statements ")
    parser.add_argument("--tcp_ip_port", type=str, help="IP:PORT of tcp server", default=None)
    parser.add_argument("--save_frames_to",default=None, type=str)
    parser.add_argument("--split_detector", action="store_true", help = "If set true, Splits the frame into 4 eqaul quadrants and aggregates the results at the end")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect_video()

 #python  pc.py --tcp_ip_port 192.168.50.1:9999 --video_path rtsp://192.168.50.1:40000 --save_frames_to frames
 #python  pc.py --tcp_ip_port 192.168.50.1:9999 --video_path rtsp://192.168.50.1:40000 --save_frames_to frame   --proc_freq 40
 #python pc.py --display --tcp_ip_port 192.168.50.1:9999 --video_path /home/rajneesh/Downloads/all_ideaf_videos/03April202012_23_22_Raw.mp4