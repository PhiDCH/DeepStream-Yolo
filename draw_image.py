import cv2
from bmw_draw import draw_on_rack_1, draw_on_rack_2, draw_on_rack_3, draw_on_rack_4, classify_bbox_on_rack_rating
import json 

rack_dict = json.load(open("./rack_information.json"))

def draw_on_image(im0, label_json):
    racks = []
    original_img = im0.copy()
    # xac dinh cac rack co trong frame
    for index, obj in enumerate(label_json):
        if obj["ObjectClassName"] in ['rack_1', 'rack_2', 'rack_3', 'rack_4']: 
            left, top, right, bottom, conf = obj['Left'], obj['Top'], obj['Right'], obj['Bottom'], obj['Conf']

            rack = {
                        'rack_type': obj["ObjectClassName"], 
                        'left': left, 
                        'top': top, 
                        'right': right, 
                        'bottom': bottom,
                        'conf': conf,
                        'top_box_info': [],
                        'bottom_box_info': [],
                        'left_box_info': [],
                        'right_box_info': [],
                        'n_full': 0,
                        'n_empty': 0
                    }
            racks.append(rack)
            # if obj["ObjectClassName"] == 'rack_1':
            #     im0 = plot_one_box([left, top, right, bottom], im0, (32, 32, 32), 'rack_1')
            # elif obj["ObjectClassName"] == 'rack_2':
            #     im0 = plot_one_box([left, top, right, bottom], im0, (0, 255, 255), 'rack_2')
            # elif obj["ObjectClassName"] == 'rack_3':
            #     im0 = plot_one_box([left, top, right, bottom], im0, (0, 128, 255), 'rack_3')
            # else:
            #     im0 = plot_one_box([left, top, right, bottom], im0, (255, 102, 255), 'rack_4')

    # xac dinh bbox klt nao trong rack nao
    for index, obj in enumerate(label_json):
        if obj["ObjectClassName"] in ['klt_box_full', 'klt_box_empty']: 

            left, top, right, bottom = obj['Left'], obj['Top'], obj['Right'], obj['Bottom']
            if (bottom-top) > 200 or (right-left) > 200 or (right-left)*(bottom-top) > 30000:
                continue
            cx = int((left + right)/2)
            # cy = int((top + bottom)/2)
            # if obj["ObjectClassName"] == 'klt_box_full':
            #     # im0 = plot_one_box([left, top, right, bottom], im0, (102, 204, 0), 'Full KLT box')
            #     im0 = plot_one_box([left, top, right, bottom], im0, (102, 204, 0), '')
            # else:
            #     # im0 = plot_one_box([left, top, right, bottom], im0, (102, 0, 102), 'Empty KLT box')
            #     im0 = plot_one_box([left, top, right, bottom], im0, (102, 0, 102), '')
            for rack in racks:
                if (cx - rack['left'])*(cx - rack['right']) <= 0:
                    rack['top_box_info'].append(top)
                    rack['bottom_box_info'].append(bottom)
                    rack['left_box_info'].append(left)
                    rack['right_box_info'].append(right)
                    if obj["ObjectClassName"] == 'klt_box_full':
                        rack['n_full'] += 1
                    else:
                        rack['n_empty'] += 1


    # xac dinh so klt trong moi shelf cua moi rack
    available_pl = []
    for rack in racks:
        if rack['left'] <= 3 or 1280 - rack['right'] <= 3:
            continue
        shelf_info = rack_dict[rack['rack_type']]['information']
        num_shelves = rack_dict[rack['rack_type']]['num_shelves']

        # num_all_placeholders = rack_dict[rack['rack_type']]['num_all_placeholders']
        # N_Pholders = num_all_placeholders-len(rack['top_box_info'])
        # rack_conf = rack['conf']
        # cen_rack = int((rack['left'] + rack['right'])/2)
        # n_full = rack['n_full']
        # n_empty = rack['n_empty']
        # im0[rack['bottom']:rack['bottom']+90, cen_rack-125:cen_rack+125] = 255
        # cv2.putText(im0, f'FPS: {30}', (cen_rack-125, rack['bottom']+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (32, 32, 32), 1, cv2.LINE_AA)        
        # cv2.putText(im0, f'rack_conf: {rack_conf}', (cen_rack-125, rack['bottom']+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (32, 32, 32), 1, cv2.LINE_AA)        
        # cv2.putText(im0, f'N_Pholders: {N_Pholders}', (cen_rack, rack['bottom']+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(im0, f"N_full_KLT: {n_full}", (cen_rack, rack['bottom']+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (102, 204, 0), 1, cv2.LINE_AA)
        # cv2.putText(im0, f'N_empty_KLT: {n_empty}', (cen_rack, rack['bottom']+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (102, 0, 102), 1, cv2.LINE_AA)


        if len(rack['top_box_info']) != 0:

            if rack['rack_type'] == 'rack_4':
                try:
                    rets, labels = classify_bbox_on_rack_rating(rack, num_shelves, shelf_info)
#                         print("----Draw rack 4----")
                    draw_on_rack_4(im0, rets, labels, rack, available_pl)
                except:
                    print("-----Draw on rack 4 FAILED-----")

            elif rack['rack_type'] == 'rack_3':
                try:
                    rets, labels = classify_bbox_on_rack_rating(rack, num_shelves, shelf_info)
#                     print("----Draw rack 3----")    
                    draw_on_rack_3(im0, rets, labels, rack, available_pl)
                except:
                    print("-----Draw on rack 3 FAILED-----")
            elif rack['rack_type'] == 'rack_2':
                try:
#                     print("----Draw rack 1----")
                    draw_on_rack_2(im0, rack, available_pl)
                except:
                    print("-----Draw on rack 2 FAILED-----")
            elif rack['rack_type'] == 'rack_1':
                try:
#                     print("----Draw rack 1----")
                    draw_on_rack_1(original_img, im0, rack, available_pl)
                except:
                    print("-----Draw on rack 1 FAILED-----")
            else:
                print("Invalid rack")

    # # draw box      
    # for index, obj in enumerate(available_pl):
    #     left, top, right, bottom = obj['left'], obj['top'], obj['right'], obj['bottom']
    #     cx = int((left + right)/2)
    #     cy = int((top + bottom)/2)
    #     lb = [left, top, right, bottom]
    #     # im0 = plot_one_box(lb, im0, (0, 0, 255), "Free Placeholder")
    #     im0 = plot_one_box(lb, im0, (0, 0, 255), "")
    return available_pl
    # im0 = cv2.putText(im0, f'FPS: {1/(t2-t0)}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


