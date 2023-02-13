import numpy as np
import cv2
from tracker2.utils import class_dict_inv
import os

class config:    
    # tracking args
    track_thresh=0.7
    track_buffer=30
    match_thresh=0.9
    name = None
    mot20 = False
    aspect_ratio_thresh = 0.9
    iou_linear_predict = 0.8
    linear_vel = 1.1
    
    
def convert_yolo(inp:list):
    np_lb = np.array(inp)
    np_lb [:, 1:5:2] *=1280
    np_lb [:, 2:5:2] *=720
    np_lb_bbox = np_lb.copy()
    np_lb_bbox[:, 1] = np_lb[:, 1] - np_lb[:, 3]/2
    np_lb_bbox[:, 2] = np_lb[:, 2] - np_lb[:, 4]/2
    np_lb_bbox[:, 3] = np_lb[:, 1] + np_lb[:, 3]/2
    np_lb_bbox[:, 4] = np_lb[:, 2] + np_lb[:, 4]/2
    return np_lb_bbox

def unpack_rack(tracked_rack):
    rack_info = {}
    for t in tracked_rack:
        x1,y1,x2,y2 = t[:4]
        tid = t[4]
        obj_cl = t[5]
        score = t[6]
        # if (y2-y1)/(x2-x1) < aspect_ratio_thresh:
        #     continue
        ktlfull_in_rack = []
        ktlempty_in_rack = []
        rack_info[tid]={'class':class_dict_inv[int(obj_cl)], 'position':t[:4].astype(int).tolist(), 'conf':score}
    return rack_info

def fill_form(track_klt,track_rack,  rack_with_klt, set_klt_rack, rack_info):
    
    for klt in track_klt:
        klt_x1, klt_y1, klt_x2, klt_y2 = klt[:4]
        if klt[4]not in set_klt_rack:
                continue
        for rack in track_rack:
            x1,y1,x2,y2 = rack[:4]
            rack_id = rack[4]
            if rack_id not in rack_with_klt:
                rack_with_klt[rack_id] = set()
            if x1<klt_x2<x2 and y1<klt_y2<y2:
                rack_with_klt[rack_id].add(klt[4])
                if 'klt' not in  rack_info[rack_id]:
                    rack_info[rack_id]['klt'] = []
                rack_info[rack_id]['klt'].append(klt.astype(int).tolist())
    return rack_with_klt, rack_info

def read_yolo_label(label_folder):
    all_label = {}
    
    for files in sorted([os.path.join(label_folder, i) for i in os.listdir(label_folder)]):
        # print(files)
        num_frame = files[:-4].split("_")[-1]
        with open(files) as f:
            labels = f.readlines()
            labels = [i.split(" ") for i in labels]
            labels = [[float(k) for k in i] for i in labels]
            all_label[num_frame] = labels
    
    return all_label

def devide(klts_boxes, track_rack):
    devided_klt = []
    for klt in klts_boxes:
        cl, klt_x1, klt_y1, klt_x2, klt_y2 = klt[:5]
        found_rack = 0
        for rack in track_rack:
            x1,y1,x2,y2 = rack[:4]
            rack_id = rack[4]
            rack_cl = rack[5]
            if x1<klt_x2<x2 and y1<klt_y2<y2:
                devided_klt.append([*klt, rack_id, rack_cl])
                found_rack = 1
                continue
        if found_rack == 0:
            devided_klt.append([*klt, -1, -1])
    return np.array(devided_klt)