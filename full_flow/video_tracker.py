import cv2
import json
from sort_tracker.sort import Sort
from tracker.byte_tracker import BYTETracker
import numpy as np
import ipdb
from utils import plot_tracking, draw, get_color, class_dict, class_dict_inv, color_dict, img_h, img_w
from tqdm import tqdm
import yaml
import os

def get_json_track_record(detect_path, source, config, typtr = "byte"):
    print("tracking ver 1")
    with open(config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    tracker_config = config['tracker']
    class Config:
        track_thresh = tracker_config['track_thresh']
        track_buffer = tracker_config['track_buffer']
        match_thresh = tracker_config['match_thresh']
        name = tracker_config['name']
        aspect_ratio_thresh = tracker_config['aspect_ratio_thresh']
        mot20 = tracker_config['mot20']
        
    args = Config()
    cap = cv2.VideoCapture(source)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = Sort(tracker_config['sort_track']['conf'], max_age = tracker_config['sort_track']['max_age'], iou_threshold = tracker_config['sort_track']['iou_threshold'])
    
    tracker_klt_full_rack = {}
    all_label = {}
    for files in sorted([os.path.join(detect_path, i) for i in os.listdir(detect_path)]):
        num_frame = files[:-4].split("_")[-1]
        with open(files) as f:
            labels = f.readlines()
            labels = [i.split(" ") for i in labels]
            labels = [[float(k) for k in i] for i in labels]
            labels.sort(key=lambda x: x[1])
            all_label[num_frame] = labels
            
    frame_id = 0
    aspect_ratio_thresh = 0.9
    frame_info = {}  # dict with rack and track klt each frame
    pbar = tqdm(total = video_length)
    set_klt_rack = {1:set(), 2:set(), 3:set(), 4:set(), 5:set(), 6:set(), 7:set(), 8:set(), 9:set(), 10:set(),11:set(), 12:set()}
    now_klt_id_set = {}
    prev_klt_id_set1 = {1:set(), 2:set(), 3:set(), 4:set(), 5:set(), 6:set(), 7:set(), 8:set(), 9:set(), 10:set(),11:set(), 12:set()}
    prev_klt_id_set2 = {1:set(), 2:set(), 3:set(), 4:set(), 5:set(), 6:set(), 7:set(), 8:set(), 9:set(), 10:set(),11:set(), 12:set()}
    prev_klt_id_set3 = {1:set(), 2:set(), 3:set(), 4:set(), 5:set(), 6:set(), 7:set(), 8:set(), 9:set(), 10:set(),11:set(), 12:set()}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id+=1
        # print("frame --------------------", frame_id)
        pbar.update(1)
        try:
            lb = all_label[str(frame_id)]
        except:
            continue
        np_lb = np.array(lb)
        np_lb [:, 1:5:2] *=1280
        np_lb [:, 2:5:2] *=720
        np_lb_bbox = np_lb.copy()
        only_klt_center_wh = np_lb_bbox[np.where(np_lb[:,0]>=4)]
        np_lb_bbox[:, 1] = np_lb[:, 1] - np_lb[:, 3]/2
        np_lb_bbox[:, 2] = np_lb[:, 2] - np_lb[:, 4]/2
        np_lb_bbox[:, 3] = np_lb[:, 1] + np_lb[:, 3]/2
        np_lb_bbox[:, 4] = np_lb[:, 2] + np_lb[:, 4]/2
        only_rack_box = np_lb_bbox[np.where(np_lb_bbox[:,0]<4)]
        only_klt_box = np_lb_bbox[np.where(np_lb_bbox[:,0]>=4)]
        online_racks = tracker.update(only_rack_box, [img_h, img_w], [img_h, img_w])
        dict_rack_full = {}
        rack_info = {}
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_racks:
            x1,y1,x2,y2 = t[:4]
            tid = t[4]
            obj_cl = t[5]
            score = t[6]
            ktlfull_in_rack = []
            ktlempty_in_rack = []
            rack_info[tid]={'class':class_dict_inv[int(obj_cl)], 'position':t[:4].astype(int).tolist(), 'conf':score}
            for ii, (cl, x, y,xx, yy, conf) in enumerate(only_klt_box):
                vertical = (xx-x)/(yy-y) > aspect_ratio_thresh
                if x1<xx<x2 and y1<yy<y2 and vertical:
                    ktlfull_in_rack.append(only_klt_box[ii])
            ktlfull_in_rack = np.array(ktlfull_in_rack)   
            dict_rack_full[tid] =  ktlfull_in_rack
#         online_im = plot_tracking(frame, online_racks[:,:4], online_racks[:,4],online_racks[:,5], color=(255,255,0), frame_id=frame_id + 1, fps=1)
        for itd in dict_rack_full:
            if itd not in tracker_klt_full_rack:
                if typtr == "byte":
                    tracker_klt_full_rack[itd] = BYTETracker(args, max_age=1000)
                else:
                    tracker_klt_full_rack[itd] = Sort(trt, max_age=1000, iou_threshold=mt)

            if len(dict_rack_full[itd]) == 0:
                continue
            online_klt = tracker_klt_full_rack[itd].update(dict_rack_full[itd], [img_h, img_w], [img_h, img_w])
            if len(online_klt)==0:
                continue
            if int(itd) >= len(set_klt_rack):
                continue

            now_klt_id_set[itd] = set(online_klt[:, 4].tolist())
            set_klt_rack[itd].update(prev_klt_id_set1[itd].intersection(now_klt_id_set[itd], prev_klt_id_set2[itd], prev_klt_id_set3[itd]))

            prev_klt_id_set3[itd] = prev_klt_id_set2[itd]
            prev_klt_id_set2[itd] = prev_klt_id_set1[itd]
            prev_klt_id_set1[itd] = now_klt_id_set[itd]

            box_klt_draw=[]
            cl_klt_draw=[]
            id_klt_draw=[]
            online_klt_keep = []
            for klt in online_klt:
                if klt[-2] in set_klt_rack[itd]:
                    online_klt_keep.append(klt)
                    box_klt_draw.append(klt[:4])
                    cl_klt_draw.append(klt[5])
                    id_klt_draw.append(klt[4])
            online_klt_keep = np.array(online_klt_keep)
            # set_klt_rack[int(itd)].update(online_klt[:, 4].tolist())
            rack_info[itd]['klt'] = online_klt_keep.astype(int).tolist()
#             online_im = plot_tracking(online_im, box_klt_draw, id_klt_draw,cl_klt_draw, color=(255,0,0), frame_id=frame_id + 1, fps=1)
        frame_info[frame_id] = rack_info
    print("\nnumber klts in each rack", [(i, len(set_klt_rack[i])) for i in set_klt_rack])
    return frame_info


