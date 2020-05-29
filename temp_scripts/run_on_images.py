from detectron2_detection import Detectron2
import cv2
import os
import glob
from tqdm import tqdm
from util import bbox_cxywh_xywh, draw_bboxes_xywh 
w_p ="/nfs/gpu14_datasets/surveillance_weights/visdrone_t1/model_0111599.pth"

cfg_p ="/nfs/gpu14_datasets/surveillance_weights/visdrone_t1/test.yaml"

det = Detectron2(cfg_path=cfg_p, weights_path=w_p)
imgs_list = glob.glob("/nfs/gpu14_datasets/client_datasets/idea_forge/annotations/flight_param_gt/*/*.jpg")
save_root = "/nfs/gpu14_datasets/client_datasets/idea_forge/annotations/model_preds"
txt_path = "{}/{}/{}.txt"
img_path = "{}/{}/{}"
for img in tqdm(imgs_list):
    img_name = os.path.basename(img)
    img_id =  ".".join(img_name.rsplit(".")[:-1])
    video_name = os.path.basename(os.path.dirname(img))
    img_dir = os.path.join(save_root, video_name)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    img_save_path = img_path.format(save_root, video_name, img_name)
    txt_save_f = open(txt_path.format(save_root, video_name, img_id), 'w')
    im = cv2.imread(img)
    bbox_xcycwh, cls_conf, cls_ids = det.detect(im)
    bbox_xywh = []
    if bbox_xcycwh is not None and bbox_xcycwh != []:    
        mask = cls_ids == 0
        bbox_xcycwh = bbox_xcycwh[mask]
        bbox_xywh = bbox_cxywh_xywh(bbox_xcycwh)
        im = draw_bboxes_xywh(im, [bbox_xywh])
    cv2.imwrite(img_save_path, im)
    f_h, f_w = im.shape[:2]
    for box in  bbox_xywh:
        x, y, w, h = box
        x, y, w, h = x/f_w, y/f_h, w/f_w, h/f_h
        txt_save_f.write("{},{},{},{}\n".format(x, y, w, h))
    txt_save_f.close()
    
    

    


