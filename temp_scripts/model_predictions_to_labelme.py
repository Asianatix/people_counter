from detectron2_detection import Detectron2
import cv2
import glob 
import os 
from tqdm import tqdm 
from util import bbox_cxywh_xywh,  draw_bboxes_xywh
import PIL
import io
import base64, codecs
import numpy as np
root = "/data/client_datasets/idea_forge/videos/"
vs = ["03April202017_42_05.mp4", "07April202012_40_03.mp4", "10April202019_02_52.mp4", "11April202016_23_55.mp4", "13April202017_22_58.mp4", "14April202009_16_45.mp4"]
vids = [root+i for i in vs]
save_root = "/data/client_datasets/idea_forge/annotations/test_set_v0.2"
w_p ="/nfs/gpu14_datasets/surveillance_weights/visdrone_t1/model_0111599.pth"
cfg_p ="/nfs/gpu14_datasets/surveillance_weights/visdrone_t1/test.yaml"
process_freq = 100
det = Detectron2(cfg_path=cfg_p, weights_path=w_p)

import json 

def get_shape_j(points):
    shape_j =  {
        "shape_type": "rectangle", 
        "points": points, # points = =[[x,y], [x,y]]
        "flags": {}, 
        "group_id": "All", 
        "label": "human"
        }
    return shape_j
def get_f_j(shapes_l):
    j_dict = dict(
        shapes = shapes_l,# [shape1, shape2]
        imagePath = "",
        flags = {},
        version = "4.4.0",
        imageHeight = 0,
        imageWidth=0,
        imageData=None
    )
    return j_dict
def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

def img_arr_to_b64(cv_img):
    image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(image)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData

for v in vids:
    print(v)
    base_name = os.path.basename(v).replace(".mp4", "_")
    save_f = os.path.join(save_root, base_name)
    os.mkdir(save_f)
    cap = cv2.VideoCapture(v)
    f_count = 0
    init_num = 100000
    num_saved =  0
    total_fr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #bar = tqdm(total=total_fr)
    
    while True:
        f, im = cap.read()
        if not f:
            break
        f_count += 1
     #,   bar.update(f_count)


        if not f_count % process_freq == 0:
            continue        
        
        bbox_xcycwh, cls_conf, cls_ids = det.detect(im)
        if bbox_xcycwh is not None and bbox_xcycwh != []:    
            mask = cls_ids == 0
            bbox_xcycwh = bbox_xcycwh[mask]
            persons_count = len(bbox_xcycwh)
            bbox_xywh = bbox_cxywh_xywh(bbox_xcycwh)                
            if persons_count > 0:
                im_save_p = os.path.join(save_f, "{}.jpg".format(init_num + f_count))
                cv2.imwrite(im_save_p, im)
                num_saved += 1
                print(num_saved, f_count, total_fr)
                img_w, img_h = im.shape[:2]
                all_shapes = []
                for box in  bbox_xywh:
                    x1,y1, w, h = box
                    x2, y2 =  x1 + w, y1 + h
                    all_shapes.append(get_shape_j([[x1, y1],[x2, y2]]))
                
                ann_j = get_f_j(all_shapes)
                ann_j["imageWidth"] =  img_w
                ann_j["imageHeight"] = img_h
                ann_j["imagePath"] =  "{}.jpg".format(init_num + f_count)              
                ann_j["imageData"] = img_arr_to_b64(im)
                ann_save_p = open(os.path.join(save_f, "{}.json".format(init_num + f_count)), "w")
                json.dump(ann_j, ann_save_p)
                
                
                        






