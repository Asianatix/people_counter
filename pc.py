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
from util import draw_bboxes, get_bbox_xywh, agrregate_split_results, draw_bboxes_xywh, VideoCapture, bbox_cxywh_xywh
import json 
from TCP.TCPClient import TCPClient
import queue, threading, time
import math 
from tqdm import tqdm 
class Detector(object):
    def __init__(self, args):
        self.args = args
        
        use_cuda = bool(strtobool(self.args.use_cuda))
        
        self.detectron2 = Detectron2(self.args.detectron_cfg, self.args.detectron_ckpt)
        if self.args.deep_sort:
            self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        
    def _set_tcp_client(self):
        ip, port = self.args.tcp_ip_port.strip().split(':')
        port = int(port)
        self.tcp_client = TCPClient(ip, port)
        self.tcp_client.LaunchConnection()

    def _set_video_writer(self, video_path):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        try:
            self.im_width = int(self.vdo.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20
        except Exception as e:
            self.im_width = 1280
            self.im_height = 786
            fps = 20
        self.video_output = cv2.VideoWriter(video_path, fourcc, fps, (self.im_width, self.im_height))
        
    def __enter__(self):
        if self.args.tcp_ip_port is not None:
            self._set_tcp_client() 
        self.vdo = VideoCapture(self.args.video_path, self.args.capture_buffer_length, real_time = self.args.real_time)
        assert self.vdo.cap.isOpened()
        self.vd_name = os.path.basename(self.args.video_path)
        if self.args.save_video_to:
            self._set_video_writer("{}/0_{}_{}".format(self.args.save_video_to,self.args.save_video_freq, self.vd_name))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    import math
    @staticmethod
    def filter_imgs_buffer(im_list, max_len = 3):
        c = math.floor(len(im_list)/2)
        filtered_ims = [im_list[0], im_list[c], im_list[-1]]
        return filtered_ims
    
    def detect_im(self, im, apply_batch_ensemble = False):
        
        if not isinstance(im, list):
            im = [im]
        
        batch_outs = self.detectron2.detect_batch(im, apply_batch_ensemble)
        final_bbox_xywh = []
        for idx, each_im_ouputs in enumerate(batch_outs):
            bbox_xcycwh, cls_conf, cls_ids = each_im_ouputs
            persons_count  = 0
            # Some objects are found 
            bbox_xywh = []
            if bbox_xcycwh is not None and bbox_xcycwh != []:    
                mask = cls_ids == 0
                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2
                cls_conf = cls_conf[mask]
                if self.args.deep_sort:
                    outputs = self.deepsort.update_new(bbox_xcycwh, cls_conf, im[idx])
                    persons_count = len(outputs)
                    if persons_count > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        bbox_xywh = get_bbox_xywh(bbox_xyxy, identities)
                        final_bbox_xywh.append(bbox_xywh)
                else:
                    bbox_xywh = bbox_cxywh_xywh(bbox_xcycwh)
                    final_bbox_xywh.append(bbox_xywh)
        return final_bbox_xywh
    
    
    def _get_latest_frames(self, filter_policy = None):
        
        flag, im = self.vdo.read()
        assert isinstance(im, list)
        if not self.args.supress_verbose:
            print("video_frame ids collected for processing: {}".format([x[0] for x in im]))
        im = [x[1] for x in im]
        if len(im) > 1:
            init_l = len(im)
            if filter_policy is not None:
                im = self.filter_imgs_buffer(im)
            if not self.args.supress_verbose:
                print("Only processing {} out of {} latest_frames".format(len(im), init_l))
        return flag, im
            
                
    def detect_video(self):    
        self.frame_count = 0
        self.processing_frame_count = 0
        persons_count = 0
        
        model_avg_time = 0.0
        model_total_time = 0.0

        proc_avg_time = 0.0
        proc_total_time = 0.0
        init_time = time.time()
        pbar = tqdm(total=self.vdo.total_frames)
        while True:
            frame_start_time = time.time()
        
            flag, imgs = self._get_latest_frames(filter_policy=self.args.buffer_filter_policy)
            self.frame_count += len(imgs)
            
            if not flag:
                break
            f_h, f_w = imgs[0].shape[:2]
            
            if (self.frame_count-1) % self.args.proc_freq == 0:
                self.processing_frame_count += len(imgs)
                model_init_time = time.time()                
                bbox_xywhs = self.detect_im(imgs, self.args.apply_batch_ensemble)
                model_end_time = time.time()
                persons_count = len(bbox_xywhs[-1])
            im = imgs[-1]
            if persons_count > 0 and (self.args.display or  self.args.save_video_to or (self.args.save_frames_to is not None)):
                
                im = draw_bboxes_xywh(im, bbox_xywhs, None)
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
                    for bbox_xywh in bbox_xywhs:
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

            pbar.update(self.vdo.fr_count)
            if not self.args.supress_verbose:
                
                model_avg_fps = self.frame_count // proc_total_time
                proc_avg_time = proc_total_time / self.frame_count
                frame_time = (time.time() - frame_start_time)
                nn_time = model_end_time - model_init_time
                actual_fps =   self.frame_count // (time.time() - init_time)
                
                video_fps = self.vdo.fr_count // (time.time() - init_time)
                cap_f_count = self.vdo.fr_count
                
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
    parser.add_argument("--real_time", action="store_true", help="If set true, Queue size of video capture buffer is set to <capture_buffer_length>.\
                                                                And nn processes gets only latest <capture_buffer_length> frames from  stream and\
                                                                     discards all frames that got captured while nn process isn't ready to accept.\
                                                                default behaviour is to  never discards any frame and\
                                                                     while nn process is ready reads the next <capture_buffer_length> ")
    parser.add_argument("--capture_buffer_length",type=int, default=1, help="Size of video capture buffer")
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--proc_freq", type=int, default=1, help="Frequency at which nn model process  frames_list captured from video capture buffer.")
    parser.add_argument("--display",  action="store_true",help="To display on cv window")
    parser.add_argument("--save_video_to", type=str, default=None, help="Save path to video")
    parser.add_argument("--save_video_freq", type=int, default=100000, help="Save video at this frequency of frames")
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--supress_verbose", action="store_true", help="Supress print statements ")
    parser.add_argument("--tcp_ip_port", type=str, help="IP:PORT of tcp server", default=None)
    parser.add_argument("--save_frames_to",default=None, type=str, help = "set this to path to save predicted frames to be saved")
    parser.add_argument("--split_detector", action="store_true", help = "<Not supported> If set true, Splits the frame into 4 eqaul quadrants and aggregates the results at the end")
    parser.add_argument("--detectron_ckpt", help="Path to detectron checkpoint", default = "/data/surveillance_weights/visdrone_t1/model_0111599.pth")
    parser.add_argument("--detectron_cfg", help ="path to detectron cfg", default = "/data/surveillance_weights/visdrone_t1/test.yaml")
    parser.add_argument("--buffer_filter_policy", type = str, default = None)
    parser.add_argument("--apply_batch_ensemble", action="store_true", help="Applies nms on predictions made on batch")
    parser.add_argument("--deep_sort", action="store_true", help="Using deep sort to track people.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    with Detector(args) as det:
        det.detect_video()

 #python  pc.py --tcp_ip_port 192.168.50.1:9999 --video_path rtsp://192.168.50.1:40000 --save_frames_to frames
 #python  pc.py --tcp_ip_port 192.168.50.1:9999 --video_path rtsp://192.168.50.1:40000 --save_frames_to frame   --proc_freq 40
 #python pc.py --display --tcp_ip_port 192.168.50.1:9999 --video_path /home/rajneesh/Downloads/all_ideaf_videos/03April202012_23_22_Raw.mp4