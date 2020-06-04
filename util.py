import numpy as np
import cv2
import queue, threading, time

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]


def draw_bbox(img, box, cls_name, identity=None, offset=(0,0)):
    '''
        draw box of an id
    '''
    x1,y1,x2,y2 = [int(i+offset[idx%2]) for idx,i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity%len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
    cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img


def draw_bboxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = COLORS_10[id%len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
        # cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        h,w = abs(y1-y2), abs(x1-x2)
        area = h*w
        cv2.putText(img,"{}_{}_{}".format(area, h, w),(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def draw_bboxes_xywh(img, bbox_xy_whs, identities = None, offset=(0, 0)):
    for bbox_xy_wh in bbox_xy_whs:
        z = np.array(bbox_xy_wh)
        z[:, 2] = z[:, 2] + z[:, 0]
        z[:, 3] = z[:, 3] + z[:, 1]
        bbox_xy_xy = z.tolist()
        for i,box in enumerate(bbox_xy_xy):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0    
            color = COLORS_10[id%len(COLORS_10)]
            label = '{}{:d}'.format("", id)
            cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
            #t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            # cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            # h,w = abs(y1-y2), abs(x1-x2)
            # area = h*w
            #cv2.putText(img,"{}_{}_{}".format(area, h, w),(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def get_bbox_xywh(bbox, identities, offset=(0, 0)):
    ret_boxes = []
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        x,y,w,h = x1, y1, abs(x2-x1), abs(y2-y1)
        ret_boxes.append([x, y, w, h])
    return ret_boxes

def bbox_cxywh_xywh(bbox):
    if len(bbox) == 0:
         return bbox
    bbox[:,0] = bbox[:,0] - bbox[:,2]/2
    bbox[:,1] = bbox[:,1] - bbox[:,3]/2
    return bbox
def softmax(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(x*5)
    return x_exp/x_exp.sum()

def softmin(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(-x)
    return x_exp/x_exp.sum()


def agrregate_split_results(split_frame_bbox_list, h_w, h_h):
    bbox_xy_wh = []
    lt_out, rt_out, lb_out, rb_out = split_frame_bbox_list
    bbox_xy_wh += lt_out
    if len(rt_out) > 0:
        rt = np.array(rt_out)
        x_c = rt[:, 0]
        x_c = x_c + h_w
        rt[:, 0] = x_c
        bbox_xy_wh += rt.tolist()

    if len(lb_out) > 0:
        lb = np.array(lb_out)
        x_c = lb[:, 1]
        x_c = x_c + h_h
        lb[:, 1] = x_c
        bbox_xy_wh += lb.tolist()
        
    if len(rb_out) > 0:
        rb = np.array(rb_out)
        x_c = rb[:, 0]
        x_c = x_c + h_w
        rb[:, 0] = x_c
        x_c = rb[:, 1]
        x_c = x_c + h_h
        rb[:, 1] = x_c
        bbox_xy_wh += rb.tolist()
    
    return bbox_xy_wh

class VideoCapture:
    def __init__(self, video_path, buffer_size = 3, real_time = True):
        self.buffer_size = buffer_size
        self.cap = cv2.VideoCapture(video_path)
        self.real_time = real_time
        if self.real_time:
            self.q = queue.Queue()
            t = threading.Thread(target=self._reader)
            t.daemon = True
            t.start()
        self.fr_count = 0
        try:
            self.total_frames = self.get(cv2.CAP_PROP_FRAME_COUNT)
        except Exception as e:
            self.total_frames = 1000000
  # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:        
            ret, frame = self.cap.read()
            if not ret:
                break
            self.fr_count += 1
            if self.q.qsize() >= self.buffer_size:
                try:
                    self.q.get_nowait()   # discard LastIn frame (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put([self.fr_count, frame])
            time.sleep(0.04)

    def read(self):
        frames_list = []
        # if real-time read and remove latest buffered frames
        if self.real_time:
            for i in range(self.buffer_size):
                try:
                    f = self.q.get()
                    frames_list.append(f)
                except queue.Empty:
                    break
        # read next buffer size frames.This might slow down capturing speed but we use it only during offline processing.
        else:
            for i in range(self.buffer_size):
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.fr_count += 1
                frames_list.append([self.fr_count, frame])
        return len(frames_list)>0, frames_list



if __name__ == '__main__':
    x = np.arange(10)/10.
    x = np.array([0.5,0.5,0.5,0.6,1.])
    y = softmax(x)
    z = softmin(x)
    import ipdb; ipdb.set_trace()