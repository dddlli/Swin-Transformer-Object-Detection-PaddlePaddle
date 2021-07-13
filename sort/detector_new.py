import paddlehub as hub
import os
import time
from extractor_new import *
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

object_detector = hub.Module(name="yolov3_darknet53_pedestrian")

def get_object_position_new(img_roi, img, min_confidence, net):
    list_ = []
    confidences = []
    img_crop = []
    result = object_detector.object_detection(images=[img], use_gpu=True, score_thresh=0.00, visualization=False)
    position = result[0]['data']
    for dict_position in position:
        if dict_position['label'] == 'pedestrian':
            x = (dict_position['left']) 
            y = (dict_position['top']) 
            x1 = (dict_position['right']) 
            y1 = (dict_position['bottom']) 
            w = x1 - x
            h = y1 - y
            if h/w <= 1:
                continue
            confidence = dict_position['confidence']
            roi = img_roi[int(y):int(y1), int(x):int(x1)]
            cv2.imwrite('deep_sort_paddle/roi_img/1.jpg', roi)
            img_crop.append(roi)

            if confidence >= min_confidence:
                p_p = (x, y, w, h) #(x, y, w, h)
                list_.append(p_p)
                confidences.append(confidence)
    time_ = time.time()
    extr = Extractor(net)
    feature = extr(img_crop)
    time_extr = time.time()
    print('extr_time:')
    print(time_extr-time_)
    return list_, confidences, feature