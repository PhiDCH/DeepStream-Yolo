# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy
import cv2
import time
from operator import *
from count_klt import *
from sklearn.cluster import DBSCAN

# with torch.cuda.device(device_number):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('tf_efficientnetv2_b2', pretrained=False)
model.reset_classifier(39)
model.load_state_dict(torch.load('PositionClassify/EffectionNet_Classifiy/saved_model/best_acc_6078.pth', map_location='cpu'))
model.eval()


def dbscan_cluster(klt_lst, klt_height_lst, distance_scale = 0.5, min_point_of_cluster = 2):
#     print(klt_lst.shape, klt_height_lst.shape)
    klt_y_values = np.array(klt_lst)
    distance = np.mean(klt_height_lst) * distance_scale

    clustering = DBSCAN(eps=distance, min_samples=min_point_of_cluster).fit(klt_y_values)
    labels = clustering.labels_
    return klt_lst, labels


def classify_rack_2_enhence(img, rack_pos, klt_pos, device_number = 0):
    device = torch.device(f'cuda:{device_number}')
    gpu_model = model.to(device)
    return count_klt_image_v2(img, rack_pos, klt_pos, device, gpu_model)


rack_place_holder_dict = {
    'rack_1': 4 * 6 * 2,
    'rack_2': 3 * 3 * 3 + 4 * 3,
    'rack_3': 3 * 1 * 2,
    'rack_4': 3 * 2 * 2 + 2 * 2 * 1
}

rack_shelf_mapping = {
    'rack_1': np.array([6 * 2, 6 * 2, 6 * 2, 6 * 2]),
    'rack_2': np.array([3 * 3, 3 * 3, 3 * 3, 4 * 3]),
    'rack_3': np.array([1 * 2, 1 * 2, 1 * 2]),
    'rack_4': np.array([2 * 2, 1 * 2, 2 * 2, 2 * 2, 1 * 2])
}

rack_shelf_num = {
    'rack_1': 4,
    'rack_2': 4,
    'rack_3': 3,
    'rack_4': 5
}

shelf_ratio_dict = {
    'rack_1': {
        0:{ 'top': 0, 'bottom': 0.27685},
        1:{ 'top': 0.27685, 'bottom': 0.44463},
        2:{ 'top': 0.44463, 'bottom': 0.59899},
        3:{ 'top': 0.59899, 'bottom': 1.0}
    },
    'rack_2': {
        0:{ 'top': 0, 'bottom': 0.24691},
        1:{ 'top': 0.24691, 'bottom': 0.54938},
        2:{ 'top': 0.54938, 'bottom': 0.7428},
        3:{ 'top': 0.7428, 'bottom': 1.0}
    },
    'rack_3': {
        0:{ 'top': 0, 'bottom': 0.07805},
        1:{ 'top': 0.07805, 'bottom': 0.34146},
        2:{ 'top': 0.34146, 'bottom': 1.0}
    },
    'rack_4': {
        0:{ 'top': 0, 'bottom': 0.15288},
        1:{ 'top': 0.15288, 'bottom': 0.33633},
        2:{ 'top': 0.33633, 'bottom': 0.53237},
        3:{ 'top': 0.53237, 'bottom': 0.71942},
        4:{ 'top': 0.71942, 'bottom': 1.0}
    }
}


# Đọc thông tin file kết quả tracking json
def get_json_tracking_content(json_tracking_path):
    if not os.path.isfile(json_tracking_path):
        raise Exception("File not exist")
    with open(json_tracking_path, 'r') as f:
        json_content = json.load(f)
    return json_content

# Lấy ra danh sách unique id rack trong
def get_rack_obj_id_set(json_content):
    rack_obj_id_set = set()
    for frame_idx, frame_track in tqdm(json_content.items()):
        for k in frame_track.keys():
            rack_obj_id_set.add(k)
    return rack_obj_id_set

# Lấy thông tin KLT trong mỗi rack
def get_rack_obj_klt_position_dict(rack_obj_id_set, json_tracking_content):
    each_rack_obj_klt_position_dict = dict()
    for rack_obj in tqdm(rack_obj_id_set):
        for frame_idx, frame_track in json_tracking_content.items():
            for k, v in frame_track.items():
                if k == rack_obj:
                    if k not in each_rack_obj_klt_position_dict.keys():
                        each_rack_obj_klt_position_dict[k] = {}
                        each_rack_obj_klt_position_dict[k]['frame'] = frame_idx
                        each_rack_obj_klt_position_dict[k]['rack_name'] = v['class']
                        each_rack_obj_klt_position_dict[k]['conf'] = []
                        each_rack_obj_klt_position_dict[k]['cont_frame'] = 0
                        each_rack_obj_klt_position_dict[k]['rack_position'] = []
                        each_rack_obj_klt_position_dict[k]['appear_frame'] = {}
                        each_rack_obj_klt_position_dict[k]['klt'] = {}
                    each_rack_obj_klt_position_dict[k]['cont_frame'] += 1
                    each_rack_obj_klt_position_dict[k]['rack_position'].extend([v['position']])
                    each_rack_obj_klt_position_dict[k]['conf'].extend([v['conf']])
                    if 'klt' not in v.keys():
                        continue
                    each_rack_obj_klt_position_dict[k]['appear_frame'][frame_idx] = len(v['klt'])
                    klt_lst = v['klt']
                   
                    for item in klt_lst:
                        try:
                            if len(item) == 7:
                                x1, y1, x2, y2, klt_id, klt_cls, _ = item
                            else:
                                x1, y1, x2, y2, klt_id, klt_cls = item
                        except:
                            print(item)
                            raise Exception('My error!')
                        if klt_id not in each_rack_obj_klt_position_dict[k]['klt'].keys():
                            each_rack_obj_klt_position_dict[k]['klt'][klt_id] = {}
                            each_rack_obj_klt_position_dict[k]['klt'][klt_id]['position'] = []
                            each_rack_obj_klt_position_dict[k]['klt'][klt_id]['bottom'] = []
                            each_rack_obj_klt_position_dict[k]['klt'][klt_id]['height'] = []
                        center_h = (y1 + y2) / 2
    #                     center_h = y2
                        each_rack_obj_klt_position_dict[k]['klt'][klt_id]['class'] = klt_cls
                        each_rack_obj_klt_position_dict[k]['klt'][klt_id]['position'].extend([center_h])
                        each_rack_obj_klt_position_dict[k]['klt'][klt_id]['bottom'].extend([y2])
                        each_rack_obj_klt_position_dict[k]['klt'][klt_id]['height'].extend([y2 - y1])
            
    return each_rack_obj_klt_position_dict

# Xử lý thông tin KLT để sinh thông tin hữu ích sử dụng để cluster shelf
def get_general_rack_obj_infomation_in_video(each_rack_obj_klt_position_dict):
    result_dict = dict()
    for rack_idx, klt_content in sorted(each_rack_obj_klt_position_dict.items()):
        result_dict[rack_idx] = {}
        result_dict[rack_idx]['klt'] = {}
        result_dict[rack_idx]['rack_name'] = klt_content['rack_name']
        result_dict[rack_idx]['conf'] = np.sum(klt_content['conf']) / klt_content['cont_frame']
        result_dict[rack_idx]['cont_frame'] = klt_content['cont_frame']
        result_dict[rack_idx]['appear_frame'] = klt_content['appear_frame']
#         result_dict[rack_idx]['rack_position'] = klt_content['rack_position']
        process_arr = np.array(klt_content['rack_position'])
        result_dict[rack_idx]['rack_median_top'] = np.median(process_arr[:,1])
        result_dict[rack_idx]['rack_median_bot'] = np.median(process_arr[:,3])
        result_dict[rack_idx]['rack_median_height'] = np.median((process_arr[:,3] - process_arr[:,1])) 
        for klt_key, klt_height_lst in klt_content['klt'].items():
            if klt_key not in result_dict[rack_idx]['klt'].keys():
                result_dict[rack_idx]['klt'][klt_key] = {}
            result_dict[rack_idx]['klt'][klt_key]['class'] = klt_height_lst['class']
            result_dict[rack_idx]['klt'][klt_key]['median_height'] = np.median(np.array(klt_height_lst['position']))
            result_dict[rack_idx]['klt'][klt_key]['bottom'] = np.median(np.array(klt_height_lst['bottom']))
            result_dict[rack_idx]['klt'][klt_key]['median_height_of_klt'] = np.median(np.array(klt_height_lst['height']))
            result_dict[rack_idx]['klt'][klt_key]['number_of_display_frame'] = len(klt_height_lst['position'])
    return result_dict

# Tiền xử lý các KLT trong rack, mục đích là loại bỏ các tracking nhầm dựa theo cluster và thông tin của rack
def pre_processing_rack_input_dict(general_rack_obj_infomation_in_video, filter_by_frame = 30):
    temp_arr = []
    for rack_id, rack_id_content in tqdm(sorted(general_rack_obj_infomation_in_video.items())):
        rack_name = rack_id_content['rack_name']
        klt_info = rack_id_content['klt']
        cont_frame = rack_id_content['cont_frame']
        if cont_frame < filter_by_frame:
            del general_rack_obj_infomation_in_video[rack_id]
            continue
        sel_arr = []
        
        sort_by_height_arr = sorted(klt_info.items(),key=lambda x:getitem(x[1],'median_height'))
        
        sel_arr = [[x[1]['median_height']] for x in sort_by_height_arr]
        n_clusters = rack_shelf_num[rack_name]
        if len(sel_arr) < n_clusters:
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.array(sel_arr))
        pre = kmeans.predict(sel_arr)
        
        tmp = deepcopy(pre)
        shelf_pholder = []
        middle_checker = 0
        anchor = pre[0]
        
        anchor_idx_lst = [0]
        for idx in range(len(tmp)):
            item = tmp[idx]
            if item == anchor:
                middle_checker += 1
            else:
                shelf_pholder.append(middle_checker)
                anchor = pre[idx]
                anchor_idx_lst.extend([idx])
                middle_checker = 1
            if idx == len(tmp) - 1:
                shelf_pholder.append(middle_checker)
                
        anchor_idx_lst.append(len(pre))
        shelf_N_Pholders = np.array(rack_shelf_mapping[rack_name]) - np.array(shelf_pholder)
        
        for idx in range(len(shelf_N_Pholders)):
            if shelf_N_Pholders[idx] < 0:
                number_remove_sample = abs(shelf_N_Pholders[idx])
                sel_area = sort_by_height_arr[anchor_idx_lst[idx]: anchor_idx_lst[idx+1]]
                sel_area = sorted(sel_area, key=lambda x: (x[1]['number_of_display_frame']))
                sel_area = sel_area[:number_remove_sample]
                for item in sel_area:
                    del_id, _ = item
                    del general_rack_obj_infomation_in_video[rack_id]['klt'][del_id]

# +
# Generate kết quả từ thông tin KLT và rack
def get_submission_eval_video_lst(general_rack_obj_infomation_in_video, json_tracking_content, video_path='', is_confidence = True):
    submission_lst = dict()
    sample_centroid = {}
    for rack_idx, rack_content in tqdm(sorted(general_rack_obj_infomation_in_video.items())):
        rack_name = rack_content['rack_name']
        conf_score = rack_content['conf']
        cont_frame = rack_content['cont_frame']
#         if conf_score < 0.95 and cont_frame < 240:
#             continue
            
        klt_info = rack_content['klt']
        sel_arr, bottom_arr, height_arr = [], [], []
        
        obj = {
            "rack_name": rack_name,
            "rack_conf": 1 if is_confidence else conf_score,
        }
        
        num_empty, num_full = 0, 0
        for k_klt, v_klt in klt_info.items():
            sel_arr.append([v_klt['median_height']])
            bottom_arr.append([v_klt['bottom']])
            height_arr.append([v_klt['median_height_of_klt']])
            if v_klt['class'] == 4:
                num_empty += 1
            else:
                num_full += 1
        obj["N_empty_KLT"] = int(num_empty)
        obj["N_full_KLT"] = int(num_full)
        obj["N_Pholders"] = int(rack_place_holder_dict[rack_name] - num_empty - num_full)
        obj["shelf_N_Pholders"] = {}
        
#         try:
        if obj["N_Pholders"] == 0:
            submission_lst[rack_idx] = obj
            continue

        if obj["N_Pholders"] == rack_place_holder_dict[rack_name]:
            obj["shelf_N_Pholders"] = {}
            for idx in range(len(rack_shelf_mapping[rack_name])):
                shelf_real_num = idx+1
                obj["shelf_N_Pholders"][f'shelf_{shelf_real_num}'] = rack_shelf_mapping[rack_name][idx]
            submission_lst[rack_idx] = obj
            continue

        sel_arr.sort()
        n_clusters = rack_shelf_num[rack_name]

#         klt_lst, labels = dbscan_cluster(bottom_arr, height_arr)
#         unique_cluster = len(np.unique(labels))
#         print(f'DBScan {rack_idx} - {rack_name} - Number of cluster: {unique_cluster}')

        if len(sel_arr) < n_clusters:
            occur_lst = [0] * rack_shelf_num[rack_name]
            rack_median_height = rack_content['rack_median_height']
            rack_median_top = rack_content['rack_median_top']
            for shelf_idx in range(len(occur_lst)):
                shelf_klt = 0
                shelf_info = shelf_ratio_dict[rack_name][shelf_idx]
                top, bottom = rack_median_top + shelf_info['top'] * rack_median_height, rack_median_top + shelf_info['bottom'] * rack_median_height
                for klt_key, klt_content in klt_info.items():
                    klt_median_height = klt_content['median_height']
#                     print(top, bottom, klt_median_height)
                    if rack_name == 'rack_3' and shelf_idx == 0:
                        if klt_median_height <= bottom:
                            shelf_klt += 1
                    else:
                        if top <= klt_median_height and klt_median_height <= bottom:
                            shelf_klt += 1
                occur_lst[shelf_idx] = shelf_klt
            shelf_N_Pholders = np.array(rack_shelf_mapping[rack_name]) - np.array(occur_lst)
            for idx in range(len(shelf_N_Pholders)):
                if shelf_N_Pholders[idx] > 0:
                    obj["shelf_N_Pholders"][f"shelf_{idx+1}"] = int(shelf_N_Pholders[idx])
            submission_lst[rack_idx] = obj
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.array(sel_arr))

        # Thêm đoạn check cluster centroid để check xem có shelf nào không có KLT không
        # ----------------------------------------------------------------------------------------------
        sample_centroid[f'{rack_idx}_{rack_name}'] = sorted(kmeans.cluster_centers_, key=lambda x: x[0])
        centroid_sorted_lst = sorted(kmeans.cluster_centers_, key=lambda x: x[0])
#         print(f'Center: {sample_centroid}')
        is_missing = False
        for centroid_idx in range(1, len(centroid_sorted_lst)):
            check_condition = centroid_sorted_lst[centroid_idx] - centroid_sorted_lst[centroid_idx-1]
            if check_condition < 60:
                is_missing = True
                break
        if is_missing:
            occur_lst = [0] * rack_shelf_num[rack_name]
            rack_median_height = rack_content['rack_median_height']
            rack_median_top = rack_content['rack_median_top']
            for shelf_idx in range(len(occur_lst)):
                shelf_klt = 0
                shelf_info = shelf_ratio_dict[rack_name][shelf_idx]
                top, bottom = rack_median_top + shelf_info['top'] * rack_median_height, rack_median_top + shelf_info['bottom'] * rack_median_height
                for klt_key, klt_content in klt_info.items():
                    klt_median_height = klt_content['median_height']
#                     print(top, bottom, klt_median_height)
                    if rack_name == 'rack_3' and shelf_idx == 0:
                        if klt_median_height <= bottom:
                            shelf_klt += 1
                    else:
                        if top <= klt_median_height and klt_median_height <= bottom:
                            shelf_klt += 1
                occur_lst[shelf_idx] = shelf_klt
            shelf_N_Pholders = np.array(rack_shelf_mapping[rack_name]) - np.array(occur_lst)
            for idx in range(len(shelf_N_Pholders)):
                if shelf_N_Pholders[idx] > 0:
                    obj["shelf_N_Pholders"][f"shelf_{idx+1}"] = int(shelf_N_Pholders[idx])
            submission_lst[rack_idx] = obj
            continue
        # ----------------------------------------------------------------------------------------------

        pre = kmeans.predict(sel_arr)
        tmp = deepcopy(pre)
        shelf_pholder = []
        middle_checker = 0
        anchor = pre[0]
        for idx in range(len(tmp)):
            item = tmp[idx]
            if item == anchor:
                middle_checker += 1
            else:
                shelf_pholder.append(middle_checker)
                anchor = pre[idx]
                middle_checker = 1
            if idx == len(tmp) - 1:
                shelf_pholder.append(middle_checker)

        shelf_N_Pholders = np.array(rack_shelf_mapping[rack_name]) - np.array(shelf_pholder)
        for idx in range(len(shelf_N_Pholders)):
                if shelf_N_Pholders[idx] > 0:
                    obj["shelf_N_Pholders"][f"shelf_{idx+1}"] = int(shelf_N_Pholders[idx])

        submission_lst[rack_idx] = obj
#         except:
#             obj["N_Pholders"] = 0
#             obj["shelf_N_Pholders"] = {}
#             submission_lst[rack_idx] = obj
    return submission_lst


# -

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# Lưu file json
def generate_json_submission(submission_eval_video_dict, mAP, FPS, submission_name):
    submission_eval_video_lst = []
    for k, v in sorted(submission_eval_video_dict.items()):
        submission_eval_video_lst.append(v)
    submission_json = {
    "eval_video" : submission_eval_video_lst,
    "mAP": float(mAP),
    "FPS": float(FPS)
    }
#     print(submission_json)
    with open(f'{submission_name}.json', 'w+') as f:
        json.dump(submission_json, f, cls = NpEncoder, indent=4)

def check_rack2_occur(json_tracking_content, video_path):
    temp_dict = {}
    result_dict = {}
    for frame_idx, frame_content in json_tracking_content.items():
        for rack_idx, rack_content in frame_content.items():
            if rack_content['class'] != 'rack_2':
                continue
                
            if rack_idx not in result_dict.keys():
                result_dict[rack_idx] = {
                'class': rack_content['class'],
                'occur': set()
            }
            if rack_idx not in temp_dict.keys():
                temp_dict[rack_idx] = []
            rack_pos = rack_content['position']
#             try:
            if 'klt' not in rack_content.keys():
                continue
            klt_pos_lst = rack_content['klt']
            klt_pred_lst = []
            for klt_pos in klt_pos_lst:
                try:
                    x1, y1, x2, y2, klt_id, cls = klt_pos
                except:
                    x1, y1, x2, y2, klt_id, cls, rackidd = klt_pos
                if klt_id not in temp_dict[rack_idx]:
                    temp_dict[rack_idx].append(klt_id)
                    klt_pred_lst.append([x1, y1, x2, y2])

            if len(klt_pred_lst) == 0:
                continue
            cap = cv.VideoCapture(video_path)
            cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
            res, frame = cap.read()
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pre_rack2_occur = classify_rack_2_enhence(image, rack_pos, klt_pred_lst)
            for item in pre_rack2_occur:
                result_dict[rack_idx]['occur'].add(item)
#             except:
#                 print(rack_content)
    print(result_dict)
    return result_dict
#     print(result_dict)


def run_full_pipeline(json_tracking_content, mAP, FPS, submission_name, video_path, is_confidence= False):
    rack_obj_id_set = get_rack_obj_id_set(json_tracking_content)
    rack_obj_klt_position_dict = get_rack_obj_klt_position_dict(rack_obj_id_set, json_tracking_content)
    general_rack_obj_infomation_in_video = get_general_rack_obj_infomation_in_video(rack_obj_klt_position_dict)
    pre_processing_rack_input_dict(general_rack_obj_infomation_in_video)
#     with open('general_rack_obj_infomation_in_video.json', 'w+') as f:
#         json.dump(general_rack_obj_infomation_in_video, f, cls = NpEncoder)
    submission_eval_video_dict = get_submission_eval_video_lst(general_rack_obj_infomation_in_video,json_tracking_content, video_path, is_confidence = True)
    rack2_occur_pos = check_rack2_occur(json_tracking_content, video_path)
    for rack_id, rack_occur in sorted(rack2_occur_pos.items()):
        occur_lst = np.array(list(rack_occur['occur']))
#         print(occur_lst)
        shelf_4_num = len(np.where(occur_lst<=11)[0])
        shelf_3_num = len(np.where((12<=occur_lst) & (occur_lst<=20))[0])
        if rack_id in submission_eval_video_dict.keys():
            if 'shelf_4' in submission_eval_video_dict[rack_id]['shelf_N_Pholders'].keys():
                npholders = 12 - shelf_4_num
                if npholders > 0:
                    submission_eval_video_dict[rack_id]['shelf_N_Pholders']['shelf_4'] = npholders
                elif npholders == 0:
                    del submission_eval_video_dict[rack_id]['shelf_N_Pholders']['shelf_4']
                    
            if 'shelf_3' in submission_eval_video_dict[rack_id]['shelf_N_Pholders'].keys():
                npholders = 9 - shelf_3_num
                if npholders > 0:
                    submission_eval_video_dict[rack_id]['shelf_N_Pholders']['shelf_3'] = npholders
                elif npholders == 0:
                    del submission_eval_video_dict[rack_id]['shelf_N_Pholders']['shelf_3']
            total_npholder = 0
            for k, v in submission_eval_video_dict[rack_id]['shelf_N_Pholders'].items():
                total_npholder += v
            submission_eval_video_dict[rack_id]['N_Pholders'] = total_npholder
            
    generate_json_submission(submission_eval_video_dict, mAP, FPS, submission_name)
#     start = time.time()
#     end = time.time()
#     print(end - start)
    print('Finish')
