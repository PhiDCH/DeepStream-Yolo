import os
import argparse
import time
from pathlib import Path

from json_generator import *
from video_tracker import *
from video_tracker_v2 import get_json_track_record_v2
# +
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection', nargs='+', type=str, default='', help='detection labels path')
    parser.add_argument('--source', nargs='+', type=str, default='', help='video path')
    parser.add_argument('--config', nargs='+', type=str, default='config.yaml', help='config path')
   
    opt = parser.parse_args()
    if opt.detection == '' or opt.source == '':
        raise Exception("Detection or video not found")
        
    with open(opt.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    general_config = config['general']
    
    json_track_record = get_json_track_record_v2(opt.detection[0], opt.source[0], opt.config)
    
    
#     json_track_record = {}
#     for idx in range(3253, 3500):
#         fake_pred = {
#             10.0: {
#                 'class': 'rack_4',
#                 'position': [928, 167, 1256, 593],
#                 'conf': 1,
#                 'klt': [[955, 135, 1035, 170, 1, 5],
#                        [955, 300, 1035, 350, 2, 5],
#                        [955, 390, 1035, 500, 3, 5]]
#             }
#         }
#         json_track_record[idx] = fake_pred
#    with open('byte_match-0.9_score-0.7_folder3_sort-rack_ver0.9_prev3_linearvel11_ioulp_0.8.json', 'r') as f:
#        json_track_record = json.load(f)
        
#    temp_json = {}
#    for k, v in json_track_record.items():
#        if k not in temp_json.keys():
#                temp_json[int(k)] = {}
#        for sub_k, sub_v in v.items():
#            if sub_k not in temp_json[int(k)].keys():
#                temp_json[int(k)][int(float(sub_k))] = {}
#            temp_json[int(k)][int(float(sub_k))] = sub_v
    mAP, FPS, submission_name = general_config['mAP'], general_config['FPS'], general_config['submission_name']
    is_confidence = general_config['is_confidence']
    run_full_pipeline(json_track_record, mAP, FPS, submission_name, opt.source[0])

#     with open('result.json', 'w+') as f:
#         json.dump(json_track_record, f)
