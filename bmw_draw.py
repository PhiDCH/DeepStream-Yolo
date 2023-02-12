
import glob
import json
from tqdm import tqdm
import random
import cv2
import numpy as np
import time
from post_process_rack_1 import absence_klt_interprete, imshow, get_subimages
from post_process_rack_2 import absence_klt_interprete_rack_2


object_list = ['rack_1', 'rack_2', 'rack_3', 'rack_4', 'klt_box_full', 'klt_box_empty']
rack_dict = json.load(open("./rack_information.json"))
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(7)]
W = 1280
H = 720


def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def plot_image(json_files):
    for json_file in tqdm(json_files):
        with open(json_file) as f:
            label_json = json.load(f)
        image_file = json_file.replace("/labels/json/", "/images/").replace(".json", ".jpg")
        img = cv2.imread(image_file)
        for index, obj in enumerate(label_json):
            left, top, right, bottom = obj['Left'], obj['Top'], obj['Right'], obj['Bottom']
            cx = int((left + right)/2)
            cy = int((top + bottom)/2)
            id_class = object_list.index(obj["ObjectClassName"])
            lb = [left, top, right, bottom]
            img = plot_one_box(lb, img, colors[id_class], obj["ObjectClassName"])
#             img = plot_one_box(lb, img, colors[id_class], str(obj["Id"]))
            if obj["ObjectClassName"] in ['klt_box_full', 'klt_box_empty']:
                img = cv2.circle(img, (cx, cy), 5, (255, 0, 0), thickness=2)
        cv2.imwrite(image_file.replace("/images/", "/sample_images/"), img)

def classify_bbox_on_rack_rating(rack, num_shelves, shelf_info):
    rets = []
    labels = []
    top_rack = rack['top']
    bottom_rack = rack['bottom']
    top_box_info = rack['top_box_info']
    bottom_box_info = rack['bottom_box_info']
    
    cnt = 0
    for i in range(len(top_box_info)):
        flag = False
        top_box = top_box_info[i]
        bottom_box = bottom_box_info[i]
        
        if top_box < top_rack:
            labels.append(1)
            flag = True
            continue
            
        h = (bottom_box + top_box)/2
        for ind in range(num_shelves):
            shelf = shelf_info[ind]
            shelf_top = shelf[f'shelf_{ind+1}']['top']*(bottom_rack - top_rack) + top_rack
            shelf_bottom = shelf[f'shelf_{ind+1}']['bottom']*(bottom_rack - top_rack) + top_rack

            if (h - shelf_top)*(h - shelf_bottom) <= 0 and bottom_box <= shelf_bottom:
                labels.append(ind + 1)
                flag = True
#                 print(top_box, bottom_box, shelf_top, shelf_bottom, flag)
                break
            if (h - shelf_top)*(h - shelf_bottom) <= 0 and 0 <= bottom_box - shelf_bottom <= 10: # 10: sai so box va cac duong phan chia shelves
                labels.append(ind + 1)
                flag = True
#                 print(top_box, bottom_box, shelf_top, shelf_bottom, flag)
                break
            if (h - shelf_top)*(h - shelf_bottom) <= 0 and bottom_box - shelf_bottom > 10:
                labels.append(ind + 2)
                flag = True
#                 print(top_box, bottom_box, shelf_top, shelf_bottom, flag)
                break
#             print(top_box, bottom_box, shelf_top, shelf_bottom, flag)
        if not flag:

            cnt += 1
  
    if len(labels) != len(top_box_info):
        print(f"ERROR! labels: {len(labels)}, box: {len(top_box_info)}/{len(bottom_box_info)}")
        return []
    else:
        for i in range(num_shelves):
            rets.append(labels.count(i+1))
        return rets, labels

def check_position_klt(img_cut):
    x = np.mean(img_cut)
    if x < 70:
        return False, x 
    else: 
        return True, x

def draw_on_rack_4(img, num_klt_each_cluster, cluster_result, rack, avai_pl_list):
    coef_ = 0.12161294 # hang so tinh offset ve shelf
    rack_4 = rack_dict["rack_4"]
    shelf_info = rack_4['information']
    num_shelves = rack_4['num_shelves']
    left_bbox = rack['left_box_info']
    top_bbox = rack['top_box_info']
    right_bbox = rack['right_box_info']
    bottom_bbox = rack['bottom_box_info']
    cen_rack_x = int((rack['left'] + rack['right'])/2)
    cen_rack_y = int((rack['top'] + rack['bottom'])/2)
    rate_w = 1 - abs(cen_rack_x - 640)/640/4.5
    rate_cen_x = 0.06317314

#     print(cluster_result)
    num_ph_shelf = [shelf_info[i][f'shelf_{i+1}']['cols'] * shelf_info[i][f'shelf_{i+1}']['rows'] \
                    for i in range(num_shelves)]
    
    avai_placeholders = [num_ph_shelf[i] - num_klt_each_cluster[i] for i in range(num_shelves)]
    
    for shelf, num_ph in enumerate(avai_placeholders):
        if num_ph > 0:    
            idxes = [i for i in range(0, len(cluster_result)) if cluster_result[i]==shelf+1]
            if shelf + 1 in [1, 3, 4]:
                if len(idxes) == 0:
                    # draw shelf-box
                    top_shelf = int(shelf_info[shelf][f'shelf_{shelf+1}']['top']*(rack['bottom'] - rack['top'])) + rack['top']
                    bottom_shelf = int(shelf_info[shelf][f'shelf_{shelf+1}']['bottom']*(rack['bottom'] - rack['top'])) + rack['top']
                    min_left_box = min(left_bbox)
                    max_right_box = max(right_bbox)
                    mean_w_box = int((sum(right_bbox) - sum(left_bbox))/len(right_bbox))

                    left_shelf = min_left_box if (min_left_box - rack['left'])/(rack['right'] - rack['left']) < 0.3 else rack['left']
                    right_shelf = max_right_box if (max_right_box - rack['left'])/(rack['right'] - rack['left']) > 0.7 else rack['right']        
                    offset_shelf_h = int(coef_*bottom_shelf) # 0.13084606 la coef duoc regression tu ti le shelf so voi rack-box

                    shelf_center_x, shelf_center_y = int((right_shelf + left_shelf)/2), int((top_shelf + bottom_shelf)/2)

                    avai_pl_list.append({'left': shelf_center_x-mean_w_box, 'top': bottom_shelf-offset_shelf_h//2, 'right': shelf_center_x, 'bottom': bottom_shelf})
                    avai_pl_list.append({'left': shelf_center_x-mean_w_box, 'top': bottom_shelf-offset_shelf_h, 'right': shelf_center_x, 'bottom': bottom_shelf-offset_shelf_h//2})
                    avai_pl_list.append({'left': shelf_center_x, 'top': bottom_shelf-offset_shelf_h//2, 'right': shelf_center_x+mean_w_box, 'bottom': bottom_shelf})
                    avai_pl_list.append({'left': shelf_center_x, 'top': bottom_shelf-offset_shelf_h, 'right': shelf_center_x+mean_w_box, 'bottom': bottom_shelf-offset_shelf_h//2})
                
                elif len(idxes) == 1:
                    left, top, right, bottom = left_bbox[idxes[0]], top_bbox[idxes[0]], right_bbox[idxes[0]], bottom_bbox[idxes[0]]
                    cen_box_x = int((left + right)/2)
                    cen_box_y = int((top + bottom)/2)
                    w_b = int((right - left)*rate_w)
                    h_b = bottom - top
                    offset_h = int(10 + 25*cen_box_y/720)
                    offset_box = int((bottom-top)/2)
                    
                    front_box, score = check_position_klt(img[bottom+2:bottom+20, cen_box_x-20:cen_box_x+20])
                    if front_box:
                        if rack['right'] - cen_box_x < cen_box_x - rack['left']:
                            avai_pl_list.append({'left': left, 'top': top-offset_h+offset_box, 'right': right, 'bottom': bottom-offset_h})
                            avai_pl_list.append({'left': left-w_b, 'top': top-offset_h+offset_box, 'right': right-w_b, 'bottom': bottom-offset_h})
                            avai_pl_list.append({'left': left-w_b, 'top': top+offset_box, 'right': right-w_b, 'bottom': bottom})
                        else:
                            avai_pl_list.append({'left': left, 'top': top-offset_h+offset_box, 'right': right, 'bottom': bottom-offset_h})
                            avai_pl_list.append({'left': left+w_b, 'top': top-offset_h+offset_box, 'right': right+w_b, 'bottom': bottom-offset_h})
                            avai_pl_list.append({'left': left+w_b, 'top': top, 'right': right+w_b, 'bottom': bottom})
                    else:
                        if rack['right'] - cen_box_x < cen_box_x - rack['left']:
                            avai_pl_list.append({'left': left, 'top': top+offset_h+offset_box, 'right': right, 'bottom': bottom+offset_h})
                            avai_pl_list.append({'left': left-w_b, 'top': top+offset_h+offset_box, 'right': right-w_b, 'bottom': bottom+offset_h})
                            avai_pl_list.append({'left': left-w_b, 'top': top+offset_box, 'right': right-w_b, 'bottom': bottom})
                        else:
                            avai_pl_list.append({'left': left, 'top': top+offset_h+offset_box, 'right': right, 'bottom': bottom+offset_h})
                            avai_pl_list.append({'left': left+w_b, 'top': top+offset_h+offset_box, 'right': right+w_b, 'bottom': bottom+offset_h})
                            avai_pl_list.append({'left': left+w_b, 'top': top+offset_box, 'right': right+w_b, 'bottom': bottom})
                        
                elif len(idxes) == 2:
                    left_1, top_1, right_1, bottom_1 = left_bbox[idxes[0]], top_bbox[idxes[0]], right_bbox[idxes[0]], bottom_bbox[idxes[0]]
                    left_2, top_2, right_2, bottom_2 = left_bbox[idxes[1]], top_bbox[idxes[1]], right_bbox[idxes[1]], bottom_bbox[idxes[1]]
                    cen_x_1 = int((left_1 + right_1)/2)
                    cen_y_1 = int((top_1 + bottom_1)/2)
                    cen_x_2 = int((left_2 + right_2)/2)
                    cen_y_2 = int((top_2 + bottom_2)/2)
                    
                    offset_h = int(10 + 25*(cen_y_1 + cen_y_2)/2/720)
                    offset_box = int((bottom_1 + bottom_2 - top_1 - top_2)/4) 
                    
                    w_b = int((right_1 - left_1 + right_2 - left_2)/2*rate_w)
                    # h_b = int(bottom_1 - top_1 + bottom_2 - top_2)/2
                    
                    if abs(cen_x_1 - cen_x_2) > 30:
                        front_box_1, score_1 = check_position_klt(img[bottom_1+2:bottom_1+20, cen_x_1-20:cen_x_1+20])
                        front_box_2, score_2 = check_position_klt(img[bottom_2+2:bottom_2+20, cen_x_2-20:cen_x_2+20])

                        if front_box_1:
                            avai_pl_list.append({'left': left_1, 'top': top_1-offset_h+offset_box, 'right': right_1, 'bottom': bottom_1-offset_h})
                        else:
                            avai_pl_list.append({'left': left_1, 'top': top_1+offset_h+offset_box, 'right': right_1, 'bottom': bottom_1+offset_h})
                        if front_box_2:
                            avai_pl_list.append({'left': left_2, 'top': top_2-offset_h+offset_box, 'right': right_2, 'bottom': bottom_2-offset_h})
                        else:
                            avai_pl_list.append({'left': left_2, 'top': top_2+offset_h+offset_box, 'right': right_2, 'bottom': bottom_2+offset_h})
                    else:
                        if rack['right'] - cen_x_1 < cen_x_1 - rack['left']:
                            avai_pl_list.append({'left': left_1-w_b, 'top': top_1+offset_box, 'right': right_1-w_b, 'bottom': bottom_1})
                            avai_pl_list.append({'left': left_2-w_b, 'top': top_2+offset_box, 'right': right_2-w_b, 'bottom': bottom_2})
                        else:
                            avai_pl_list.append({'left': left_1+w_b, 'top': top_1+offset_box, 'right': right_1+w_b, 'bottom': bottom_1})
                            avai_pl_list.append({'left': left_2+w_b, 'top': top_2+offset_box, 'right': right_2+w_b, 'bottom': bottom_2})
                
                elif len(idxes) == 3:
                    left_1, top_1, right_1, bottom_1 = left_bbox[idxes[0]], top_bbox[idxes[0]], right_bbox[idxes[0]], bottom_bbox[idxes[0]]
                    left_2, top_2, right_2, bottom_2 = left_bbox[idxes[1]], top_bbox[idxes[1]], right_bbox[idxes[1]], bottom_bbox[idxes[1]]
                    left_3, top_3, right_3, bottom_3 = left_bbox[idxes[2]], top_bbox[idxes[2]], right_bbox[idxes[2]], bottom_bbox[idxes[2]]
                    cen_x_1 = int((left_1 + right_1)/2)
                    cen_y_1 = int((top_1 + bottom_1)/2)
                    cen_x_2 = int((left_2 + right_2)/2)
                    cen_y_2 = int((top_2 + bottom_2)/2)
                    cen_x_3 = int((left_3 + right_3)/2)
                    cen_y_3 = int((top_3 + bottom_3)/2)
                    
                    offset_h = int(10 + 25*(cen_y_1 + cen_y_2 + cen_y_3)/3/720)
                    # offset_box = int((bottom_1 + bottom_2 + bottom_3 - top_1 - top_2 - top_3)/6)
                    offset_box = int(max([bottom_1-top_1, bottom_2-top_2, bottom_3-top_3])/2)
                    off_x1 = int((640-cen_x_1)*rate_cen_x)
                    off_x2 = int((640-cen_x_2)*rate_cen_x)
                    off_x3 = int((640-cen_x_3)*rate_cen_x)
                    
                    w_b = int((right_1 - left_1 + right_2 - left_2 + right_3 - left_3)/3*rate_w)
                    # h_b = int(bottom_1 - top_1 + bottom_2 - top_2 + bottom_3 - top_3)/3
                    
                    pos1 = rack['right'] + rack['left'] - 2*cen_x_1
                    pos2 = rack['right'] + rack['left'] - 2*cen_x_2
                    pos3 = rack['right'] + rack['left'] - 2*cen_x_3

                    if (pos1 <= 0 and pos2 <= 0 and pos3 >= 0) or (pos1 >= 0 and pos2 >= 0 and pos3 <= 0):
                        flag3, score3 = check_position_klt(img[bottom_3+2:bottom_3+20, cen_x_3-10:cen_x_3+10])
                        if flag3:
                            avai_pl_list.append({'left': left_3+off_x3, 'top': top_3-offset_h+offset_box, 'right': right_3+off_x3, 'bottom': bottom_3-offset_h})
                        else:
                            avai_pl_list.append({'left': left_3-off_x3, 'top': top_3+offset_h+offset_box, 'right': right_3-off_x3, 'bottom': bottom_3+offset_h})
                    
                    elif (pos1 <= 0 and pos2 >= 0 and pos3 <= 0) or (pos1 >= 0 and pos2 <= 0 and pos3 >= 0):
                        flag2, score2 = check_position_klt(img[bottom_2+2:bottom_2+20, cen_x_2-15:cen_x_2+15])
                        if flag2:
                            avai_pl_list.append({'left': left_2+off_x2, 'top': top_2-offset_h+offset_box, 'right': right_2+off_x2, 'bottom': bottom_2-offset_h})
                        else:
                            avai_pl_list.append({'left': left_2-off_x2, 'top': top_2+offset_h+offset_box, 'right': right_2-off_x2, 'bottom': bottom_2+offset_h})
                    
                    elif (pos1 >= 0 and pos2 <= 0 and pos3 <= 0) or (pos1 <= 0 and pos2 >= 0 and pos3 >= 0):
                        flag1, score1 = check_position_klt(img[bottom_1+2:bottom_1+20, cen_x_1-15:cen_x_1+15])
                        if flag1:
                            avai_pl_list.append({'left': left_1+off_x1, 'top': top_1-offset_h+offset_box, 'right': right_1+off_x1, 'bottom': bottom_1-offset_h})
                        else:
                            avai_pl_list.append({'left': left_1-off_x1, 'top': top_1+offset_h+offset_box, 'right': right_1-off_x1, 'bottom': bottom_1+offset_h})
                    
                    else:
                        print("No exist this case")
                     
                else:
                    print('Number of placeholders is out of range')
            
            if shelf + 1 in [2, 5]:
             
                if len(idxes) == 0:
                    # draw shelf-box
                    top_shelf = int(shelf_info[shelf][f'shelf_{shelf+1}']['top']*(rack['bottom'] - rack['top'])) + rack['top']
                    bottom_shelf = int(shelf_info[shelf][f'shelf_{shelf+1}']['bottom']*(rack['bottom'] - rack['top'])) + rack['top']
                    max_left_box = min(left_bbox)
                    max_right_box = max(right_bbox)
                    mean_w_box = int((sum(right_bbox) - sum(left_bbox))/len(left_bbox))

                    left_shelf = max_left_box if (max_left_box - rack['left'])/(rack['right'] - rack['left']) < 0.3 else rack['left']
                    right_shelf = max_right_box if (max_right_box - rack['left'])/(rack['right'] - rack['left']) > 0.7 else rack['right']        
                    offset_shelf_h = int(coef_*bottom_shelf) # coef_ la coef duoc regression tu ti le shelf so voi rack-box

                    shelf_center_x, shelf_center_y = int((right_shelf + left_shelf)/2), int((top_shelf + bottom_shelf)/2)
                    # neu la vung den hay vi tri trong shelf
                    flag_1, score_1 = check_position_klt(img[shelf_center_y:shelf_center_y+8, shelf_center_x-35:shelf_center_x])
                   
                    flag_2, score_2 = check_position_klt(img[shelf_center_y-8:shelf_center_y, shelf_center_x-35:shelf_center_x])
                    
                    flag_3, score_3 = check_position_klt(img[shelf_center_y:shelf_center_y+8, shelf_center_x:shelf_center_x+35])
                  
                    flag_4, score_4 = check_position_klt(img[shelf_center_y-8:shelf_center_y, shelf_center_x:shelf_center_x+35])
                   

                    if not flag_1:
                        avai_pl_list.append({'left': shelf_center_x-mean_w_box, 'top': bottom_shelf-offset_shelf_h//2, 'right': shelf_center_x, 'bottom': bottom_shelf})
                    
                    if not flag_2:
                        avai_pl_list.append({'left': shelf_center_x-mean_w_box, 'top': bottom_shelf-offset_shelf_h, 'right': shelf_center_x, 'bottom': bottom_shelf-offset_shelf_h//2})
                    
                    if not flag_3:
                        avai_pl_list.append({'left': shelf_center_x, 'top': bottom_shelf-offset_shelf_h//2, 'right': shelf_center_x+mean_w_box, 'bottom': bottom_shelf})
                    
                    if not flag_4:
                        avai_pl_list.append({'left': shelf_center_x, 'top': bottom_shelf-offset_shelf_h, 'right': shelf_center_x+mean_w_box, 'bottom': bottom_shelf-offset_shelf_h//2})
                
                elif len(idxes) == 1:
                    left, top, right, bottom = left_bbox[idxes[0]], top_bbox[idxes[0]], right_bbox[idxes[0]], bottom_bbox[idxes[0]]
                    cen_box_x = int((left + right)/2)
                    cen_box_y = int((top + bottom)/2)
                    w_b = int((right - left)*rate_w)
                    h_b = bottom - top
                    offset_h = int(10 + 25*cen_box_y/720)
                    offset_box = int((bottom - top)/2)
                    front_box, score = check_position_klt(img[bottom+2:bottom+20, cen_box_x-20:cen_box_x+20])

                    if front_box:
                        if rack['right'] - cen_box_x < cen_box_x - rack['left']:
                            flag, score = check_position_klt(img[bottom-15:bottom, left-20:left])
                            # flag: False -> horizontal, True -> vertical
                            if not flag:
                                avai_pl_list.append({'left': left-w_b, 'top': top+offset_box, 'right': right-w_b, 'bottom': bottom})
                            else:
                                avai_pl_list.append({'left': left, 'top': top-offset_h+offset_box, 'right': right, 'bottom': bottom-offset_h})

                        else:
                            flag, score = check_position_klt(img[bottom-15:bottom, right:right+20])
                   
                            if not flag:
                                avai_pl_list.append({'left': left+w_b, 'top': top+offset_box, 'right': right+w_b, 'bottom': bottom})
                            else:
                                avai_pl_list.append({'left': left, 'top': top-offset_h+offset_box, 'right': right, 'bottom': bottom-offset_h})
                    else:
                        if rack['right'] - cen_box_x < cen_box_x - rack['left']:
                            flag, score = check_position_klt(img[bottom-15:bottom, left-20:left])
                      
                            if not flag:
                                avai_pl_list.append({'left': left-w_b, 'top': top+offset_box, 'right': right-w_b, 'bottom': bottom})
                            else:
                                avai_pl_list.append({'left': left, 'top': top+offset_h+offset_box, 'right': right, 'bottom': bottom+offset_h})

                        else:
                            flag, score = check_position_klt(img[bottom-15:bottom, right:right+20])
            
                            if not flag:
                                avai_pl_list.append({'left': left+w_b, 'top': top+offset_box, 'right': right+w_b, 'bottom': bottom})
                            else:
                                avai_pl_list.append({'left': left, 'top': top+offset_h+offset_box, 'right': right, 'bottom': bottom+offset_h})
                        
                else:
                    print('Number of placeholders is out of range')
                    
    return avai_pl_list

def draw_on_rack_3(img, num_klt_each_cluster, cluster_result, rack, avai_pl_list):
    rack_3 = rack_dict["rack_3"]
    shelf_info = rack_3['information']
    num_shelves = rack_3['num_shelves']
    left_bbox = rack['left_box_info']
    top_bbox = rack['top_box_info']
    right_bbox = rack['right_box_info']
    bottom_bbox = rack['bottom_box_info']
    cen_rack_x = int((rack['left'] + rack['right'])/2)
    # cen_rack_y = int((rack['top'] + rack['bottom'])/2)
    coef_ = 0.12161294 # hang so tinh offset ve shelf
    rate_cen_x = 0.06317314
    
    mean_w_box = int((sum(right_bbox) - sum(left_bbox))/len(left_bbox))
    check_horizontal_rack = True if mean_w_box/(rack['right']-rack['left']) < 0.75 else False
    
    rate_w = 1 - abs(cen_rack_x - 640)/640/3 if cen_rack_x <= 640 else 1.0

    num_ph_shelf = [shelf_info[i][f'shelf_{i+1}']['cols'] * shelf_info[i][f'shelf_{i+1}']['rows'] for i in range(num_shelves)]
    
    avai_placeholders = [num_ph_shelf[i] - num_klt_each_cluster[i] for i in range(num_shelves)]
  
    for shelf, num_ph in enumerate(avai_placeholders):
        if num_ph > 0:
            idxes = [i for i in range(0, len(cluster_result)) if cluster_result[i]==shelf+1]
            
            if len(idxes) == 0:
                top_shelf = int(shelf_info[shelf][f'shelf_{shelf+1}']['top']*(rack['bottom'] - rack['top'])) + rack['top']
                bottom_shelf = int(shelf_info[shelf][f'shelf_{shelf+1}']['bottom']*(rack['bottom'] - rack['top'])) + rack['top']
                min_left_box = min(left_bbox)
                max_right_box = max(right_bbox)

                left_shelf = min_left_box if (min_left_box - rack['left'])/(rack['right'] - rack['left']) < 0.3 else rack['left']
                right_shelf = max_right_box if (max_right_box - rack['left'])/(rack['right'] - rack['left']) > 0.7 else rack['right']
                shelf_center_x, shelf_center_y = int((right_shelf + left_shelf)/2), int((top_shelf + bottom_shelf)/2)
                offset_shelf_h = int(coef_*bottom_shelf) # 0.13084606 la coef duoc regression tu ti le shelf so voi rack-box
                off_x = int((640-shelf_center_x)*rate_cen_x)
                if not check_horizontal_rack: # vertical - truong hop rack quay doc va chi nhin thay 1 cot tren shelf                  
                    avai_pl_list.append({'left': shelf_center_x-mean_w_box//2, 'top': bottom_shelf-offset_shelf_h//2, 'right': shelf_center_x+mean_w_box//2, 'bottom': bottom_shelf})
                    avai_pl_list.append({'left': shelf_center_x-mean_w_box//2+off_x, 'top': bottom_shelf-offset_shelf_h, 'right': shelf_center_x+mean_w_box+off_x, 'bottom': bottom_shelf-offset_shelf_h//2})
                else:
                    avai_pl_list.append({'left': shelf_center_x-mean_w_box, 'top': bottom_shelf-offset_shelf_h, 'right': shelf_center_x, 'bottom': bottom_shelf})
                    avai_pl_list.append({'left': shelf_center_x, 'top': bottom_shelf-offset_shelf_h, 'right': shelf_center_x+mean_w_box-off_x, 'bottom': bottom_shelf})
                    
            elif len(idxes) == 1:
                left, top, right, bottom = left_bbox[idxes[0]], top_bbox[idxes[0]], right_bbox[idxes[0]], bottom_bbox[idxes[0]]
                cen_box_x = int((left + right)/2)
                cen_box_y = int((top + bottom)/2)
                w_b = right - left
                # h_b = bottom - top
                offset_h = int(10 + 20*cen_box_y/720)
                offset_box = int((bottom - top)/2)
                off_x = int((640-cen_box_x)*rate_cen_x)
                
                front_box, score = check_position_klt(img[bottom+2:bottom+20, cen_box_x-20:cen_box_x+20])
                
                if not check_horizontal_rack: # vertical - truong hop rack quay doc va chi nhin thay 1 cot tren shelf
                    if front_box:
                        avai_pl_list.append({'left': left+off_x, 'top': top-offset_h+offset_box, 'right': right+off_x, 'bottom': bottom-10})
                    else:
                        avai_pl_list.append({'left': left-off_x, 'top': top+offset_h+offset_box, 'right': right-off_x, 'bottom': bottom+10})
                else:
                    if rack['right'] - cen_box_x < cen_box_x - rack['left']:
                        avai_pl_list.append({'left': left-w_b*rate_w, 'top': top+offset_box, 'right': right-w_b*rate_w-10, 'bottom': bottom})
                    else:
                        avai_pl_list.append({'left': left+w_b*rate_w, 'top': top+offset_box, 'right': right+w_b*rate_w-10, 'bottom': bottom})
            else:
                print('Number of placeholders is out of range')

    return avai_pl_list

def draw_on_rack_2(im0, rack, available_pl):
    existed_klts = []
    for klt_id in range(len(rack['left_box_info'])):
        existed_klts.append([
            klt_id, 
            {
                "Right": rack['right_box_info'][klt_id],
                "Top": rack['top_box_info'][klt_id],
                "Left": rack['left_box_info'][klt_id],
                "Bottom": rack['bottom_box_info'][klt_id],
        }
            ])
    rack_label = [rack['rack_type'],{
        "Right": rack['right'],
        "Top": rack['top'],
        "Left": rack['left'],
        "Bottom": rack['bottom'],
    }]

    
    available_pholders = absence_klt_interprete_rack_2(rack_label, existed_klts)
    for ap in available_pholders:
        available_pl.append({
            'top' : ap[1]['Top'],
            'left': ap[1]['Left'],
            'right': ap[1]['Right'],
            'bottom': ap[1]['Bottom']
        })
    im0 = get_subimages(im0, available_pholders)
    return im0

def draw_on_rack_1(original_img, im0, rack, available_pl):
    existed_klts = []
    for klt_id in range(len(rack['left_box_info'])):
        existed_klts.append([
            klt_id, 
            {
                "Right": rack['right_box_info'][klt_id],
                "Top": rack['top_box_info'][klt_id],
                "Left": rack['left_box_info'][klt_id],
                "Bottom": rack['bottom_box_info'][klt_id],
        }
            ])
    
    available_pholders = absence_klt_interprete(existed_klts, original_img)
    for ap in available_pholders:
        available_pl.append({
            'top' : ap[1]['Top'],
            'left': ap[1]['Left'],
            'right': ap[1]['Right'],
            'bottom': ap[1]['Bottom']
        })
    im0 = get_subimages(im0, available_pholders)
    return im0

def plot_available_placeholders_on_image(json_files):
    for json_file in tqdm(json_files):
        json_id = json_file.split("/")[-1]
        with open(json_file) as f:
            label_json = json.load(f)
        image_file = json_file.replace("/labels/json/", "/images/").replace(".json", ".jpg")
        # image_file = json_file.replace(".json", ".png")
        img = cv2.imread(image_file)
        st = time.time()
        racks = []
        # xac dinh cac rack co trong frame
        for index, obj in enumerate(label_json):
            if obj["ObjectClassName"] in ['rack_1', 'rack_2', 'rack_3', 'rack_4']: 
                left, top, right, bottom = obj['Left'], obj['Top'], obj['Right'], obj['Bottom']
                rack = {
                            'rack_type': obj["ObjectClassName"], 
                            'left': left, 
                            'top': top, 
                            'right': right, 
                            'bottom': bottom, 
                            'top_box_info': [],
                            'bottom_box_info': [],
                            'left_box_info': [],
                            'right_box_info': []
                        }
                racks.append(rack)
                if obj["ObjectClassName"] == 'rack_1':
                    img = plot_one_box([left, top, right, bottom], img, (32, 32, 32), 'rack_1')
                elif obj["ObjectClassName"] == 'rack_2':
                    img = plot_one_box([left, top, right, bottom], img, (0, 255, 255), 'rack_2')
                elif obj["ObjectClassName"] == 'rack_3':
                    img = plot_one_box([left, top, right, bottom], img, (0, 128, 255), 'rack_3')
                else:
                    img = plot_one_box([left, top, right, bottom], img, (255, 102, 255), 'rack_4')

        # xac dinh bbox klt nao trong rack nao
        for index, obj in enumerate(label_json):
            if obj["ObjectClassName"] in ['klt_box_full', 'klt_box_empty']: 
                left, top, right, bottom = obj['Left'], obj['Top'], obj['Right'], obj['Bottom']
                cx = int((left + right)/2)
                # cy = int((top + bottom)/2)
                if obj["ObjectClassName"] == 'klt_box_full':
                    img = plot_one_box([left, top, right, bottom], img, (102, 204, 0), 'Full KLT box')
                else:
                    img = plot_one_box([left, top, right, bottom], img, (102, 0, 102), 'Empty KLT box')
                for rack in racks:
                    if (cx - rack['left'])*(cx - rack['right']) <= 0:
                        rack['top_box_info'].append(top)
                        rack['bottom_box_info'].append(bottom)
                        rack['left_box_info'].append(left)
                        rack['right_box_info'].append(right)

        # xac dinh so klt trong moi shelf cua moi rack
        available_pl = []
       
        for rack in racks:
            if abs((rack['left'] + rack['right'])/2 - 640) > 550:
                continue
            if rack['left'] <= 3 or 1280 - rack['right'] <= 3:
                continue
            shelf_info = rack_dict[rack['rack_type']]['information']
            num_shelves = rack_dict[rack['rack_type']]['num_shelves']
            
            if len(rack['top_box_info']) != 0:
                
                rets, labels = classify_bbox_on_rack_rating(rack, num_shelves, shelf_info)
           
                print(f"{rack['rack_type']} on {json_id}: {rets}")
                if rack['rack_type'] == 'rack_4':
                    print("----Draw rack 4----")
                    
                    draw_on_rack_4(img, rets, labels, rack, available_pl)
                   
                if rack['rack_type'] == 'rack_3':
                    print("----Draw rack 3----")
                   
                    draw_on_rack_3(img, rets, labels, rack, available_pl)
                   
        
        # draw box      
        for index, obj in enumerate(available_pl):
            left, top, right, bottom = obj['left'], obj['top'], obj['right'], obj['bottom']
            cx = int((left + right)/2)
            cy = int((top + bottom)/2)
            lb = [left, top, right, bottom]
            img = plot_one_box(lb, img, (0, 0, 255), "Free Placeholder")
        
        en = time.time()
        print(f"Time for drawing: {en-st}, FPS: {1/(en-st)}")

        cv2.imshow("window", img)
        if cv2.waitKey(1) == ord('q'):
            break
        else:
            time.sleep(0.01)
        # cv2.imwrite(image_file.replace("/images/", "/sample_images/"), img)
        # cv2.imwrite(image_file.replace("/test/", "/sample_images/"), img)

if __name__ == "__main__":
    # json_file1 = "../datasets/Hackathon_Stage2/Evaluation_set/dataset/test/887_1.json"
    # json_file2 = "../datasets/Hackathon_Stage2/Evaluation_set/dataset/test/887_2.json"
    # json_file3 = "../datasets/Hackathon_Stage2/Evaluation_set/dataset/test/887_3.json"
    # json_file4 = "../datasets/Hackathon_Stage2/Evaluation_set/dataset/test/887_4.json"
    # json_file5 = "../datasets/Hackathon_Stage2/Evaluation_set/dataset/test/887_5.json"
    # json_file6 = "../datasets/Hackathon_Stage2/Evaluation_set/dataset/test/887_6.json"
    # json_file7 = "../datasets/Hackathon_Stage2/Evaluation_set/dataset/test/887_7.json"
    # json_file_list = [json_file1, json_file2, json_file3, json_file4, json_file5, json_file6, json_file7]
    # st = time.time()
    # plot_available_placeholders_on_image(json_file_list)
    # print(f"time: {(time.time() - st)/7}")
    import natsort
    json_files = natsort.natsorted(glob.glob("../datasets/Hackathon_Stage2/Evaluation_set/dataset/labels/json/*.json"))
    # json_file = "../datasets/Hackathon_Stage2/Evaluation_set/dataset/labels/json/652.json"
    plot_available_placeholders_on_image(json_files)
