import cv2
import json
from tracker2.sort import Sort
from tracker2.byte_tracker import BYTETracker
import numpy as np
from tracker2.utils import plot_tracking, img_h, img_w
from tqdm import tqdm
from tracker2.helper import config, convert_yolo, unpack_rack, fill_form, read_yolo_label, devide

def tracking(np_lb_bbox, prev_klt_id1, prev_klt_id2, prev_klt_id3, set_klt_rack, rack_with_klt, tracker_rack, tracker_klt_full_rack):
    only_rack_box = np_lb_bbox[np.where(np_lb_bbox[:,0]<4)]
    only_klt_box = np_lb_bbox[np.where(np_lb_bbox[:,0]>=4)]
    
    only_klt_box_filter = []
    for klt in only_klt_box:
        x1,y1,x2,y2 = klt[1:5]
        if (x2-x1)/(y2-y1) > 0.9 and x1>5:
            only_klt_box_filter.append(klt)
    only_klt_box_filter = np.array(only_klt_box_filter)
    
    
    online_racks = tracker_rack.update(only_rack_box, [img_h, img_w], [img_h, img_w])
    only_klt_box_devide = devide(only_klt_box_filter, online_racks)
    online_klt, strack_pools = tracker_klt_full_rack.update(only_klt_box_devide, [img_h, img_w], [img_h, img_w])
    rack_info = unpack_rack(online_racks)
    now_klt_id = set(online_klt[:, 4].tolist())
    set_klt_rack.update(now_klt_id.intersection(prev_klt_id1, prev_klt_id2, prev_klt_id3))
        
    prev_klt_id3 = prev_klt_id2
    prev_klt_id2 = prev_klt_id1
    prev_klt_id1 = now_klt_id
    pred_box, box_id1 = None, None
    
    rack_with_klt, rack_info = fill_form(online_klt,online_racks, rack_with_klt, set_klt_rack, rack_info)
    return rack_info, online_racks, online_klt, prev_klt_id1, prev_klt_id2, prev_klt_id3, set_klt_rack, rack_with_klt

def get_json_track_record_v2(detect_path, source, config_, typtr = "byte"):
    print("tracking ver 2")

    args = config()
    frame_info = {}  
    label_folder = detect_path
    kk = 3
    cap = cv2.VideoCapture(source)
    all_label =  read_yolo_label(label_folder)
    # import ipdb
    # ipdb.set_trace()
    # tracker_rack = Sort(0.95, max_age = 300, iou_threshold = 0.3) # config yolov7
    tracker_rack = Sort(0.7, max_age = 1000, iou_threshold = 0.1) # config yolov4
    
    tracker_klt_full_rack = BYTETracker(args, max_age=1000)
    pbar = tqdm(total = 3253)
    frame_info = {}
    rack_with_klt = {}
    prev_klt_id1, prev_klt_id2, prev_klt_id3, now_klt_id, set_klt_rack = set(), set(), set(), set(), set()
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id+=1
        pbar.update(1)
        try:
            lb = all_label[str(frame_id)]
        except:
            continue
        np_lb_bbox = convert_yolo(lb)
        
        rack_info, online_racks, online_klt, prev_klt_id1, prev_klt_id2, prev_klt_id3, set_klt_rack, rack_with_klt = tracking(np_lb_bbox, prev_klt_id1, prev_klt_id2, prev_klt_id3, set_klt_rack, rack_with_klt, tracker_rack, tracker_klt_full_rack)
        frame_info[frame_id] = rack_info
    print("\nnumber klts in each rack", [(i, len(rack_with_klt[i])) for i in rack_with_klt])
    return frame_info

if __name__ =="__main__":
    get_json_track_record_v2("/home/robotic/Downloads/deepstream/DeepStream-Yolo/output", "/home/robotic/Downloads/deepstream/DeepStream-Yolo/eval_video_1.mp4", "")