import base64
import codecs
import glob
import io
import json
import os

import cv2
import numpy as np
import PIL

from util import bbox_cxywh_xywh


def _get_shape_j(points, label="human"):
    shape_j = {
        "shape_type": "rectangle",
        "points": points,  # points = =[[x,y], [x,y]]
        "flags": {},
        "group_id": "All",
        "label": label,
    }
    return shape_j


def _get_labelme_template(im, shapes_l, imgp, imgH, imgW):

    j_dict = dict(
        shapes=shapes_l,  # [shape1, shape2]
        imagePath=imgp,
        flags={},
        version="4.4.0",
        imageHeight=imgH,
        imageWidth=imgW,
        imageData=img_arr_to_b64(im),
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
    img_pil.save(f, format="PNG")
    data = f.getvalue()
    encData = codecs.encode(data, "base64").decode()
    encData = encData.replace("\n", "")
    return encData


def get_labelme_json(imp, im, bbox, labels=None, type="xywh"):
    """
    imp: base name of the image. Save the annotation with the same basename
    im: cv image
    bbox: NX4 numpy array
    type: "xywh"/"cxywh"/"xyxy"
    labels: array of class labels in text


    Returns a json in labelme format
    """
    w, h = im.shape[:2]
    if type == "xywh":
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    elif type == "cxywh":
        bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
        bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]

    elif type == "xyxy":
        pass
    shape_jsons_l = []
    for idx, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        point = [[x1, y1], [x2, y2]]
        if labels is not None:
            label = labels[idx]
        # Default label
        else:
            label = "human"
        shape_jsons_l.append(_get_shape_j(point, label))

    return _get_labelme_template(im, shape_jsons_l, imp, h, w)


if __name__ == "__main__":
    from detectron2_detection import Detectron2
    import json

    w_p = "/nfs/gpu14_datasets/surveillance_weights/visdrone_t1/model_0111599.pth"
    cfg_p = "/nfs/gpu14_datasets/surveillance_weights/visdrone_t1/test.yaml"
    det = Detectron2(cfg_path=cfg_p, weights_path=w_p)

    ### Params to be changed
    process_freq = 100
    root = "/data/client_datasets/idea_forge/videos/29_may/"
    # vs = ["03April202017_42_05.mp4", "07April202012_40_03.mp4", "10April202019_02_52.mp4", "11April202016_23_55.mp4", "13April202017_22_58.mp4", "14April202009_16_45.mp4"]
    vs = ["06April202012_06_22.mp4"]
    vids = [root + i for i in vs]
    save_root = "/data/client_datasets/idea_forge/annotations/model_preds"

    for v in vids:
        print(v)
        base_name = os.path.basename(v).replace(".mp4", "_")
        save_f = os.path.join(save_root, base_name)
        os.mkdir(save_f)
        cap = cv2.VideoCapture(v)
        f_count = 0
        init_num = 100000
        num_saved = 0
        total_fr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while True:
            f, im = cap.read()
            if not f:
                print("Video processing completed")
                break
            f_count += 1
            if not f_count % process_freq == 0:
                continue
            bbox_xcycwh, cls_conf, cls_ids = det.detect(im)
            if bbox_xcycwh is not None and bbox_xcycwh != []:
                mask = cls_ids == 0
                bbox_xcycwh = bbox_xcycwh[mask]
                persons_count = len(bbox_xcycwh)
                if persons_count > 0:
                    im_save_p = os.path.join(
                        save_f, "{}.jpg".format(init_num + f_count)
                    )
                    cv2.imwrite(im_save_p, im)
                    num_saved += 1
                    print(num_saved, f_count, total_fr)
                    ann_j = get_labelme_json(
                        os.path.basename(im_save_p),
                        im,
                        bbox_xcycwh,
                        labels=None,
                        type="cxywh",
                    )
                    ann_save_p = open(
                        os.path.join(save_f, "{}.json".format(init_num + f_count)), "w"
                    )
                    json.dump(ann_j, ann_save_p)
