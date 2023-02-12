from copy import deepcopy
import cv2 as cv
# import matplotlib.pyplot as plt
import numpy as np
import json
from time import time
import collections
from sklearn.cluster import DBSCAN


def imshow(im):
    plt.imshow(im)
    plt.show()

def get_subimages(im_, boxes):
    im = im_.copy()
    for box_com in boxes:
        box_pos_id, box = box_com
        start_point = (box['Left'], box['Top'])
        end_point = (box['Right'], box['Bottom'])
        im = cv.rectangle(im, start_point, end_point, color = (255, 0, 0), thickness =2)
        # im = cv.putText(im, str(box_pos_id), start_point, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    
    return im

def draw_image(image, box):

    if isinstance(box, dict):
        box_trbl = [box['Top'], box['Right'], box['Bottom'], box['Left']]
    else:
        box_trbl = box    
    a = image.copy()
    start_point = (box_trbl[3], box_trbl[0])
    end_point = (box_trbl[1], box_trbl[2])
    imshow(cv.rectangle(a, start_point, end_point, color = (255, 0, 0), thickness =2))

def check_intersect(box, rack, label):
    if label['ObjectClassName'][:4] == 'rack':
        return False
    
    box_trbl = [label['Top'], label['Right'], label['Bottom'], label['Left']]
    rack_trbl = [rack['Top'], rack['Right'], rack['Bottom'], rack['Left']]
    # draw_image(image, box_trbl) 
    if (box_trbl[1] > rack_trbl[3] and box_trbl[1] < rack_trbl[1]) or \
        (box_trbl[3] > rack_trbl[3] and box_trbl[3] < rack_trbl[1]):
        x_match = True
    else:
        x_match = False
    if (box_trbl[2] > rack_trbl[0] and box_trbl[2] < rack_trbl[2]) or \
        (box_trbl[0] > rack_trbl[0] and box_trbl[0] < rack_trbl[2]):
        y_match = True
    else:
        y_match = False
    if x_match and y_match:
        return True
    else:
        return False
    return True

def klt_get(direction, ref_label, shelf_distance = 0, tray_distance = 0):
    global angel_offset_w
    if direction == 'left':
        width = ref_label['Right'] - ref_label['Left']
        res_label = deepcopy(ref_label)
        res_label['Right'] = int(res_label['Right'] - width*(1 - angel_offset_w))
        res_label['Left'] = int(res_label['Left'] - width*(1 - angel_offset_w))
        return res_label
    
    elif direction == 'right':
        width = ref_label['Right'] - ref_label['Left']
        res_label = deepcopy(ref_label)
        res_label['Right'] = int(res_label['Right'] + width*(1 - angel_offset_w))
        res_label['Left'] = int(res_label['Left'] + width*(1 - angel_offset_w))
        return res_label
    elif direction == 'down':
        res_label = deepcopy(ref_label)
        res_label['Top'] = int(res_label['Top'] + shelf_distance[0])
        res_label['Bottom'] = int(res_label['Bottom'] + shelf_distance[0])

        res_label['Right'] = int(res_label['Right'] + shelf_distance[1])
        res_label['Left'] = int(res_label['Left'] + shelf_distance[1])
        return res_label
    elif direction == 'up':
        res_label = deepcopy(ref_label)
        res_label['Top'] = int(res_label['Top'] - shelf_distance[0])
        res_label['Bottom'] = int(res_label['Bottom'] - shelf_distance[0])

        res_label['Right'] = int(res_label['Right'] - shelf_distance[1])
        res_label['Left'] = int(res_label['Left'] - shelf_distance[1])
        return res_label
    else:
        print('ERR')

def get_center_klt(existed_klts, rack_label, large_image):
    def Euclid(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    image_center =  large_image.shape[0]/2,  large_image.shape[1]/2
    center_klt = existed_klts[0]
    min_dis = 9999999999
    for klt in existed_klts:
        box_center = klt[1]['Top']/2 + klt[1]['Right']/2, klt[1]['Right']/2 + klt[1]['Left']/2
        Edistance = Euclid(image_center, box_center ) 
        if Edistance < min_dis:
            min_dis = Edistance
            center_klt = klt

        # if
    return center_klt

def get_all_row_klt(klts_, rack_label, large_image, processed_ids = [], existed_ids = []):
    klts = deepcopy(klts_)
    for klt in klts_:
        pos_id, label = klt
        if pos_id in processed_ids:
            continue
        print(pos_id)
        processed_ids.append(pos_id)
        if pos_id % 3 == 0:
            if pos_id + 1 not in existed_ids:
                existed_ids.append(pos_id + 1)
                right_klt = klt_get('right', label)
                klts.append([pos_id + 1, right_klt])
                print(f'append: {pos_id + 1}')
                stat, klts, processed_ids = get_all_row_klt(klts, rack_label, large_image, processed_ids, existed_ids)
                pass
        if pos_id % 3 == 1:
            if pos_id + 1 not in existed_ids:
                existed_ids.append(pos_id + 1)
                right_klt = klt_get('right', label)
                klts.append([pos_id + 1, right_klt])
                print(f'append: {pos_id + 1}')
                stat, klts, processed_ids = get_all_row_klt(klts, rack_label, large_image, processed_ids, existed_ids)

            if pos_id - 1 not in existed_ids:
                existed_ids.append(pos_id - 1)
                right_klt = klt_get('left', label)
                klts.append([pos_id -1, right_klt])
                print(f'append: {pos_id -1}')
                stat, klts, processed_ids = get_all_row_klt(klts, rack_label, large_image, processed_ids, existed_ids)
        if pos_id % 3 == 2:
            if pos_id - 1 not in existed_ids:
                existed_ids.append(pos_id - 1)
                right_klt = klt_get('left', label)
                klts.append([pos_id -1, right_klt])
                print(f'append: {pos_id - 1}')
                stat, klts, processed_ids = get_all_row_klt(klts, rack_label, large_image, processed_ids, existed_ids)
    if klts == klt:
        return False, klts, []
    else:
        return True, klts, processed_ids

def find_same_col_klts(klts):
    res = []
    for i, klt in enumerate(klts):
        for klt_j in klts[i:]:
            if klt[0] <3 or klt_j[0] < 3:
                continue
            if klt[0] == klt_j[0]:
                continue
            if (klt[0] - klt_j[0])%9 == 0 :
                res.append([klt, klt_j])

    return res

def get_shelf_distance(klts):
    same_col_klt = find_same_col_klts(klts)
    if len(same_col_klt) == 0:
        return -1
    distances = []
    for pair in same_col_klt:
        distance = np.abs((pair[0][1]['Top'] - pair[1][1]['Top'])/((pair[0][0] - pair[1][0])/9)), (pair[0][1]['Right'] - pair[1][1]['Right'])/((pair[0][0] - pair[1][0])/9)
        distances.append(distance)
    return np.mean(np.array(distances), axis = 0)

def get_all_column_klt(klts_, rack_label, large_image, processed_ids = [], existed_id = []):
    klts = deepcopy(klts_)
    shelf_distance = get_shelf_distance(klts)
    for klt in klts_:
        pos_id, label = klt
        if pos_id in processed_ids:
            continue
        print(pos_id)
        processed_ids.append(pos_id)
        if pos_id < 3:
            continue

        if pos_id < 12:
            if pos_id + 9 not in existed_id:
                existed_id.append(pos_id + 9)
                upper_klt = klt_get('up', label, shelf_distance=shelf_distance)
                klts.append([pos_id + 9, upper_klt])
                print(f'append: {pos_id + 9}')
                stat, klts, processed_ids = get_all_column_klt(klts, rack_label, large_image, processed_ids, existed_id= existed_id)
        elif pos_id < 29:
            if pos_id + 9 not in existed_id:
                existed_id.append(pos_id + 9)
                upper_klt = klt_get('down', label, shelf_distance=shelf_distance)
                klts.append([pos_id + 9, upper_klt])
                print(f'append: {pos_id + 9}')
                stat, klts, processed_ids = get_all_column_klt(klts, rack_label, large_image, processed_ids, existed_id= existed_id)

            
            if pos_id - 9 not in existed_id:
                existed_id.append(pos_id - 9)
                upper_klt = klt_get('left', label)
                klts.append([pos_id -9, upper_klt])
                print(f'append: {pos_id -9}')
                stat, klts, processed_ids = get_all_column_klt(klts, rack_label, large_image, processed_ids, existed_id= existed_id)

        else:
            if pos_id - 9 not in existed_id:
                existed_id.append(pos_id - 9)
                upper_klt = klt_get('left', label)
                klts.append([pos_id -9, upper_klt])
                print(f'append: {pos_id -9}')
                stat, klts, processed_ids = get_all_column_klt(klts, rack_label, large_image, processed_ids, existed_id= existed_id)


    if klts == klt:
        return False, klts, []
    else:
        return True, klts, processed_ids

def get_cluster_centers(cluster_centers):
    centers = []
    for key in cluster_centers.keys():
        if key == -1:
            continue
        cluster_points = cluster_centers[key]
        centers.append([key, np.mean(cluster_points), np.max(cluster_points) - np.min(cluster_points) ])
    return centers

def clusters_to_klts(clusters):
    out_cluster = []
    for key in clusters.keys():
        for klt in clusters[key]:
            out_cluster.append([key, klt])
            
    return out_cluster

def klts_to_list(klts):
    '''
    #cluster, Left          , Top           ,Right          , Bottom
    # '''
    out_klts = []
    for klt in klts:
        out_klts.append([
            #cluster, Left          , Top           ,Right          , Bottom
            klt[0], klt[1]['Left'], klt[1]['Top'], klt[1]['Right'], klt[1]['Bottom']

        ])
    return out_klts

def kltslist_to_klts(klts_list):
    out_klts = []
    for klt in klts_list:
        out_klts.append([
            klt[0],
            {
            "Left": klt[1],
            "Top": klt[2],
            "Right": klt[3],
            "Bottom": klt[4],
        }

        ])
    return out_klts

def klt_format_changer(klts):
    out_klts = []
    for i, klt in enumerate(klts):
        out_klts.append([i, {
            "Right": klt[2],
            "Top": klt[1],
            "Left": klt[0],
            "Bottom": klt[3],
        }])
    return out_klts

def intersect_area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

def check_exist(cklt, klts):
    from collections import namedtuple
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    klt_area = (cklt[1]-cklt[3])*(cklt[2] - cklt[4])
    rec_a = Rectangle(cklt[1], cklt[2], cklt[3], cklt[4])
    for klt in klts:
        rec_b = Rectangle(klt[1], klt[2], klt[3], klt[4])
        if intersect_area(rec_a, rec_b)/klt_area > 0.4:
            return True

    return False

def vertical_clustering(klts, distance_scale, min_samples):
    # Vertical clustering
    klt_widths = []
    klt_x_values = []
    for klt in klts:
        klt_x_values.append([0, klt[1]['Left']])
        klt_widths.append(klt[1]['Right'] - klt[1]['Left'])
    distance = np.mean(klt_widths)*distance_scale
    mean_klt_width = np.mean(klt_widths)
    clustering = DBSCAN(eps=distance, min_samples=min_samples).fit(klt_x_values)
    return clustering.labels_, mean_klt_width

def horizontal_clustering(klts, distance_scale, min_samples):
    klt_y_values = []
    klt_heights = []
    if len(klts) == 0:
        return klts
    for klt in klts:
        klt_y_values.append([0, klt[1]['Bottom']])
        klt_heights.append(klt[1]['Bottom'] - klt[1]['Top'])
    
    distance = np.mean(klt_heights)*distance_scale
    mean_klt_height = np.mean(klt_heights)

    clustering = DBSCAN(eps=distance, min_samples=min_samples).fit(klt_y_values)
    return clustering.labels_, mean_klt_height

def absence_klt_interprete_o(exist_klt, large_image, distance_scale = 0.1, min_samples = 2):

    labels,klt_mean_height = horizontal_clustering(exist_klt, distance_scale, min_samples)
    clusters = collections.defaultdict(list)
    cluster_centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        exist_klt[i][0] = label
        clusters[label].append(exist_klt[i][1])
        cluster_centers[label].append(exist_klt[i][1]['Bottom'])
    
    cluster_centers = np.array(get_cluster_centers(cluster_centers))
    cluster_centers = cluster_centers[cluster_centers[:, 1].argsort()]
    cluster_distance = np.mean(cluster_centers[:, 1][1:] - cluster_centers[:, 1][:-1] )
    clusters.pop(-1)
    klts = clusters_to_klts(clusters)

    


    # Interprete
    klts_list = np.array(klts_to_list(klts))
    min_height, max_height = np.max(klts_list[:, 4]), np.min(klts_list[:, 4])

    new_klts = klts_list.tolist()
    time_start = time()
    for klt in klts_list:
        klt_shelf_id, klt_left, klt_top, klt_right, klt_bottom = klt

        bottom_distance =  min_height - klt_bottom
        if  bottom_distance> klt_mean_height/2:
            for iscale in range(round(bottom_distance/cluster_distance)):
                below_klt = [klt[0], klt[1], int(klt[2] + (iscale + 1)*cluster_distance), klt[3], int(klt[4] + (iscale + 1)*cluster_distance)]
                if not check_exist(below_klt, new_klts):
                    new_klts.append(below_klt)
                    imshow(get_subimages(large_image*0, kltslist_to_klts(new_klts) ))

        
        top_distance = klt_bottom -max_height
        if top_distance > klt_mean_height/2:
            for iscale in range(round(top_distance/cluster_distance)):
                below_klt = [klt[0], klt[1], int(klt[2] - (1 + iscale)*cluster_distance), klt[3], int(klt[4] - (1 + iscale)*cluster_distance)]
                if not check_exist(below_klt, new_klts):
                    new_klts.append(below_klt)
                    imshow(get_subimages(large_image*0, kltslist_to_klts(new_klts) ))


        end = 1
    print(f'Interprete time: {time() - time_start}')


    
    # imshow(get_subimages(large_image, klts))
    return kltslist_to_klts(new_klts)

def get_cluster_detail(items, labels):
    clusters = collections.defaultdict(list)
    cluster_centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        items[i][0] = label
        clusters[label].append(items[i][1])
        cluster_centers[label].append(items[i][1]['Bottom'])
    
    cluster_centers = np.array(get_cluster_centers(cluster_centers))
    cluster_centers = cluster_centers[cluster_centers[:, 1].argsort()]
    cluster_distance = np.mean(cluster_centers[:, 1][1:] - cluster_centers[:, 1][:-1] )
    return clusters, cluster_centers, cluster_distance

def draw_line(xcluster_details, ycluster_details, image):
    H, W = image.shape[:2]
    for key in xcluster_details.keys():
        right, left = xcluster_details[key]['Right'], xcluster_details[key]['Left']
        color = np.random.rand(3)
        plt.plot((right, right), (0, H), color=color, linewidth=2)
        plt.plot((left, left), (0, H), color=color, linewidth=2)
    
    for key in ycluster_details.keys():
        color = np.random.rand(3)

        right, left = ycluster_details[key]['Top'], ycluster_details[key]['Bottom']
        plt.plot( (0, W), (right, right), color=color, linewidth=2)
        plt.plot((0, W),  (left, left),  color=color, linewidth=2)
    plt.imshow(image)
    plt.show()

def check_klt_insider(large_image, new_klt):
    klt_mean_color = [148, 48, 124]
    klt_Top = new_klt[1]['Top']
    klt_Bottom = new_klt[1]['Bottom']
    klt_Left = new_klt[1]['Left']
    klt_Right = new_klt[1]['Right']
    klt_h, klt_w = klt_Bottom - klt_Top, klt_Right - klt_Left
    sample_image = large_image[klt_Top + int(klt_h/3):klt_Bottom - int(klt_h/3), klt_Left + int(klt_w/3):klt_Right - int(klt_w/3)]
    sample_image_means = sample_image.reshape(-1,sample_image.shape[-1]).mean(0)
    if np.abs(sample_image_means[0]-klt_mean_color[0]) < 25 and np.abs(sample_image_means[1]-klt_mean_color[1]) < 25 and np.abs(sample_image_means[2]-klt_mean_color[2]) < 25:
        return True
    return False

def get_klt_offset(klt_center, image_shape, klt_mean_height):
    image_center = np.array([image_shape[0]/2, image_shape[1]/2])
    (klt_center[1] - image_center[1])/image_shape[1]*klt_mean_height
    return klt_mean_height/4, (klt_center[1] - image_center[1])/image_shape[1]*klt_mean_height

def get_behind_klts(new_klts, large_image, klt_mean_height):
    behind_klts = []
    for new_klt in new_klts:
        t1 = time()
        klt_mean_color = [148, 48, 124]
        klt_Top = new_klt[1]['Top']
        klt_Bottom = new_klt[1]['Bottom']
        klt_Left = new_klt[1]['Left']
        klt_Right = new_klt[1]['Right']
        klt_h, klt_w = klt_Bottom - klt_Top, klt_Right - klt_Left
        sample_image = large_image[klt_Top + int(klt_h/3):klt_Bottom - int(klt_h/3), klt_Left + int(klt_w/3):klt_Right - int(klt_w/3)]
        sample_image_means = sample_image.reshape(-1,sample_image.shape[-1]).mean(0)
        if np.abs(sample_image_means[0]-klt_mean_color[0]) < 25 and np.abs(sample_image_means[1]-klt_mean_color[1]) < 25 and np.abs(sample_image_means[2]-klt_mean_color[2]) < 25:
            continue
        klt_offset = get_klt_offset([(klt_Top + klt_Bottom)/2,(klt_Left + klt_Right)/2], large_image.shape[:2], klt_mean_height)
        behind_klt = {
                    'Right': int(new_klt[1]['Right'] - klt_offset[1]),
                    'Top': int(new_klt[1]['Top'] - klt_offset[0]),
                    'Left': int(new_klt[1]['Left'] - klt_offset[1]),
                    'Bottom': int(new_klt[1]['Bottom'] - klt_offset[0])
                }
        behind_klts.append([new_klt[0] + 'b', behind_klt])
    return behind_klts

def get_available_holders(klts, xcluster_details, ycluster_details):
    existed_klt = []
    for klt in klts:
        existed_klt.append([klt[1]['xcluster'], klt[1]['ycluster']])
    
    new_klts = []
    for xcluster_id in xcluster_details.keys():
        for ycluster_id in ycluster_details.keys():
            if [xcluster_id, ycluster_id] not in existed_klt:
                new_klt = {
                    'Right': int(xcluster_details[xcluster_id]['Right']),
                    'Top': int(ycluster_details[ycluster_id]['Top']),
                    'Left': int(xcluster_details[xcluster_id]['Left']),
                    'Bottom': int(ycluster_details[ycluster_id]['Bottom'])
                }
                new_klts.append([ f'{xcluster_id}-{ycluster_id}', new_klt])
    return new_klts

def get_current_klts(exist_klt, distance_scale, min_samples):
    ts_1 = time()
    shelf = {}

    ## Shelf Clustering
    ylabels, klt_mean_height = horizontal_clustering(exist_klt, distance_scale, min_samples)
    clusters = collections.defaultdict(list)
    yclustersdict_list = collections.defaultdict(list)
    for i, label in enumerate(ylabels):
        exist_klt[i][1]['ycluster'] = label
        clusters[label].append(exist_klt[i][1])
        yclustersdict_list[label].append([exist_klt[i][1]['Left'], exist_klt[i][1]['Top'], exist_klt[i][1]['Right'], exist_klt[i][1]['Bottom']])
    yclustersdict_list.pop(-1)
    ycluster_details = {}
    ycluster_bottoms = []
    for cluster_id in yclustersdict_list.keys():
        ycluster_details[cluster_id] = {
            'Top': np.mean(np.array(yclustersdict_list[cluster_id])[:, 1]),
            'Bottom': np.mean(np.array(yclustersdict_list[cluster_id])[:, 3]),
        }
        ycluster_bottoms.append([cluster_id, ycluster_details[cluster_id]['Bottom']])
    ts_2 = time()
    clusters.pop(-1)

    ycluster_bottoms = np.array(ycluster_bottoms)
    ycluster_bottoms = ycluster_bottoms[ycluster_bottoms[:, 1].argsort()]
    if len(ycluster_details) > 4:
        if ycluster_bottoms[1][1] - ycluster_bottoms[0][1] < klt_mean_height/2:
            ycluster_details.pop(ycluster_bottoms[0][0])

        for i in range(len(ycluster_details) - 4):
            for cluster_id in range(1, len(ycluster_bottoms) - 1):
                if (ycluster_bottoms[cluster_id][1] - ycluster_bottoms[cluster_id-1][1]) < klt_mean_height and (ycluster_bottoms[cluster_id+1][1] - ycluster_bottoms[cluster_id][1]) < klt_mean_height :
                    if ycluster_bottoms[cluster_id][0] in ycluster_details.keys():
                        ycluster_details.pop(ycluster_bottoms[cluster_id][0])

    klts = clusters_to_klts(clusters)


    ### Columns Clustering
    xlabels, klt_mean_width = vertical_clustering(klts, 0.3, 1)
    clusters = collections.defaultdict(list)
    xclustersdict_list = collections.defaultdict(list)
    for i, label in enumerate(xlabels):
        klts[i][1]['xcluster'] = label
        clusters[label].append(klts[i][1])
        xclustersdict_list[label].append([klts[i][1]['Left'], klts[i][1]['Top'], klts[i][1]['Right'], klts[i][1]['Bottom']])
    
    xcluster_details = {}
    for cluster_id in xclustersdict_list.keys():
        xcluster_details[cluster_id] = {
            'Right': np.mean(np.array(xclustersdict_list[cluster_id])[:, 2]),
            'Left': np.mean(np.array(xclustersdict_list[cluster_id])[:, 0])
        }
    return clusters, xcluster_details, ycluster_details, klt_mean_height, klt_mean_width

def Euclid(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def absence_klt_interprete(exist_klt, large_image, distance_scale = 0.1, min_samples = 2):
    if len(exist_klt) == 0:
        return exist_klt
    
    ts_1 = time()
    shelf = {}

    ## Shelf Clustering
    ylabels, klt_mean_height = horizontal_clustering(exist_klt, distance_scale, min_samples)
    clusters = collections.defaultdict(list)
    yclustersdict_list = collections.defaultdict(list)
    for i, label in enumerate(ylabels):
        exist_klt[i][1]['ycluster'] = label
        clusters[label].append(exist_klt[i][1])
        yclustersdict_list[label].append([exist_klt[i][1]['Left'], exist_klt[i][1]['Top'], exist_klt[i][1]['Right'], exist_klt[i][1]['Bottom']])
    yclustersdict_list.pop(-1)
    ycluster_details = {}
    ycluster_bottoms = []
    for cluster_id in yclustersdict_list.keys():
        ycluster_details[cluster_id] = {
            'Top': np.mean(np.array(yclustersdict_list[cluster_id])[:, 1]),
            'Bottom': np.mean(np.array(yclustersdict_list[cluster_id])[:, 3]),
        }
        ycluster_bottoms.append([cluster_id, ycluster_details[cluster_id]['Bottom']])
    ts_2 = time()
    clusters.pop(-1)

    ycluster_bottoms = np.array(ycluster_bottoms)
    ycluster_bottoms = ycluster_bottoms[ycluster_bottoms[:, 1].argsort()]
    if len(ycluster_details) > 4:
        if ycluster_bottoms[1][1] - ycluster_bottoms[0][1] < klt_mean_height/2 and ycluster_bottoms[0][0] in ycluster_details.keys():
            ycluster_details.pop(ycluster_bottoms[0][0])

        for i in range(len(ycluster_details) - 4):
            for cluster_id in range(1, len(ycluster_bottoms) - 1):
                if (
                    (ycluster_bottoms[cluster_id][1] - ycluster_bottoms[cluster_id-1][1]) < klt_mean_height 
                    and 
                    (ycluster_bottoms[cluster_id+1][1] - ycluster_bottoms[cluster_id][1]) < klt_mean_height 
                ):
                    if ycluster_bottoms[cluster_id][0] in ycluster_details.keys():
                        ycluster_details.pop(ycluster_bottoms[cluster_id][0])
    if len(ycluster_details) > 4:
        if ycluster_bottoms[1][1] - ycluster_bottoms[0][1] < klt_mean_height/2 and ycluster_bottoms[0][0] in ycluster_details.keys():
            ycluster_details.pop(ycluster_bottoms[0][0])

        for i in range(len(ycluster_details) - 4):
            for cluster_id in range(1, len(ycluster_bottoms) - 1):
                if (
                    (ycluster_bottoms[cluster_id+1][1] - ycluster_bottoms[cluster_id][1]) < klt_mean_height 
                ):
                    if ycluster_bottoms[cluster_id][0] in ycluster_details.keys():
                        ycluster_details.pop(ycluster_bottoms[cluster_id][0])

    klts = clusters_to_klts(clusters)


    ### Columns Clustering
    xlabels, klt_mean_width = vertical_clustering(klts, 0.3, 1)
    clusters = collections.defaultdict(list)
    xclustersdict_list = collections.defaultdict(list)
    for i, label in enumerate(xlabels):
        klts[i][1]['xcluster'] = label
        clusters[label].append(klts[i][1])
        xclustersdict_list[label].append([klts[i][1]['Left'], klts[i][1]['Top'], klts[i][1]['Right'], klts[i][1]['Bottom']])
    
    xcluster_details = {}
    for cluster_id in xclustersdict_list.keys():
        xcluster_details[cluster_id] = {
            'Right': np.mean(np.array(xclustersdict_list[cluster_id])[:, 2]),
            'Left': np.mean(np.array(xclustersdict_list[cluster_id])[:, 0])
        }
    klts = clusters_to_klts(clusters)

    existed_klt = []
    for klt in klts:
        existed_klt.append([klt[1]['xcluster'], klt[1]['ycluster']])
    
    available_pholders = []
    for xcluster_id in xcluster_details.keys():
        for ycluster_id in ycluster_details.keys():
            if [xcluster_id, ycluster_id] not in existed_klt:
                new_klt = {
                    'Right': int(xcluster_details[xcluster_id]['Right']),
                    'Top': int(ycluster_details[ycluster_id]['Top']),
                    'Left': int(xcluster_details[xcluster_id]['Left']),
                    'Bottom': int(ycluster_details[ycluster_id]['Bottom'])
                }
                available_pholders.append([ f'{xcluster_id}-{ycluster_id}', new_klt])

    behind_klts = []
    for new_klt in available_pholders:
        t1 = time()
        klt_mean_color = [148, 48, 124]
        klt_Top = new_klt[1]['Top']
        klt_Bottom = new_klt[1]['Bottom']
        klt_Left = new_klt[1]['Left']
        klt_Right = new_klt[1]['Right']
        klt_h, klt_w = klt_Bottom - klt_Top, klt_Right - klt_Left
        sample_image = large_image[klt_Top + int(klt_h/3):klt_Bottom - int(klt_h/3), klt_Left + int(klt_w/3):klt_Right - int(klt_w/3)]
        sample_image_means = sample_image.reshape(-1,sample_image.shape[-1]).mean(0)
        if np.abs(sample_image_means[0]-klt_mean_color[0]) < 25 and np.abs(sample_image_means[1]-klt_mean_color[1]) < 25 and np.abs(sample_image_means[2]-klt_mean_color[2]) < 25:
            continue
        klt_offset = get_klt_offset([(klt_Top + klt_Bottom)/2,(klt_Left + klt_Right)/2], large_image.shape[:2], klt_mean_height)
        behind_klt = {
                    'Right': int(new_klt[1]['Right'] - klt_offset[1]),
                    'Top': int(new_klt[1]['Top'] - klt_offset[0]),
                    'Left': int(new_klt[1]['Left'] - klt_offset[1]),
                    'Bottom': int(new_klt[1]['Bottom'] - klt_offset[0])
                }
        behind_klts.append([new_klt[0] + 'b', behind_klt])

    available_pholders = available_pholders + behind_klts
    return available_pholders


def main():
    IMAGE_PATH = 'Data/Eval/images/'
    JSON_PATH = 'Data/Eval/labels/json/'
    SAVE_PATH = 'Data/DrawVideo/from_train/'
    SAMPLING_FREQ = 1

    video_path = '/home/duongnh/Codes/BMW_SORDI/Stage2/Data/Eval/eval_video_1.mp4'
    with open('./test_DBSCAN/result.json', 'r') as f:
            label = json.load(f)
    ax1 = plt.subplot(1,1,1)
    im1 = ax1.imshow(np.ones((3, 3, 3)))
    for frame_index in range(101, 5000, 5):
        for key in label[str(frame_index)].keys():
            if 'rack_1' == label[str(frame_index)][key]['class']: # co rack 1

                cap = cv.VideoCapture(video_path)
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
                res, frame = cap.read()
                image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                
                if 'klt' not in label[str(frame_index)][key].keys():
                    continue
                klts = klt_format_changer(label[str(frame_index)][key]['klt'])
                rack_box = label[str(frame_index)][key]['position']
                rack_center = (rack_box[0]+rack_box[2])/2, (rack_box[1]+rack_box[3])/2
                if Euclid(rack_center, (image.shape[1]/2, image.shape[0]/2 )) < image.shape[1]/4: 
                    time_start = time()
                    new_klts = absence_klt_interprete(klts, image)
                    print(f'Frame {frame_index}, fps: {1/(time() - time_start)}')
                    im1.set_data(get_subimages(image, new_klts))
                else:
                    im1.set_data(image)

                plt.pause(0.001)
                
                

                plt.ioff() # due to infinite loop, this gets never called.
                plt.show()

# if __name__ == "__main__":
#     main()
    


    

          
        