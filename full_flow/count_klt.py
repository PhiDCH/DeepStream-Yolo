from copy import deepcopy
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import json
from time import time

from torchvision import datasets, transforms
import torch, timm
from torch.autograd import Variable

test_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def imshow(im):
    plt.imshow(im)
    plt.show()

def combination_1(im, mask, c= 0):
    im2 = im.copy()
    im2[:, :, c] = mask[:, :, c]
    return im2

def combination_2(im, mask, c= 0):
    im2 = im.copy()
    # mask = mask.astype(float)
    # mask[np.where(mask == 0)] = -1
    im2[:, :, c] = np.maximum(im2[:, :, c], mask[:, :, c])
    return im2

def get_subimage(im, box, rack = {}):
    black_image = im*0
    white_image = black_image + 255
    start_point = (box['Left'], box['Top'])
    end_point = (box['Right'], box['Bottom'])
    box_mask = cv.rectangle(black_image.copy(), start_point, end_point, color = (255, 255, 255), thickness =-1)
    blend_image = combination_1(im, box_mask)
    if len(rack) == 0:
        return None, blend_image
    else:
        return blend_image[rack['Top']:rack['Bottom'],rack['Left']:rack['Right']], blend_image

def get_subimage_v2(im, box, rack = [], W = 1280, H = 720):
    black_image = im*0
    white_image = black_image + 255
    x1, y1, x2, y2 = rack
    if x1 < 0:
        x1 = 0
    if x1 > W:
        x1 = W
    if y1 < 0:
        y1 = 0
    if y1 > H:
        y1 = H
    if x2 < 0:
        x2 = 0
    if x2 > W:
        x2 = W
    if y2 < 0:
        y2 = 0
    if y2 > H:
        y2 = H
    
    if (x2 - x1) == 0 or (y2 - y1) == 0:
        return None, blend_image
        
    start_point = (box[0], box[1])
    end_point = (box[2], box[3])
    box_mask = cv.rectangle(black_image.copy(), start_point, end_point, color = (255, 255, 255), thickness =-1)
    blend_image = combination_1(im, box_mask)
    if len(rack) == 0:
        return None, blend_image
    else:
        return blend_image[y1:y2,x1:x2], blend_image


def get_subimages(im_, boxes):
    im = im_.copy()
    for box_com in boxes:
        box_pos_id, box = box_com
        start_point = (box['Left'], box['Top'])
        end_point = (box['Right'], box['Bottom'])
        im = cv.rectangle(im, start_point, end_point, color = (255, 255, 255), thickness =1)
        im = cv.putText(im, str(box_pos_id), start_point, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv.LINE_AA)
    
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

def get_class(subimages, model, device, use_batch = True):
    class_mapping = { 0:0, 1:1, 2:10, 3:11, 4:12, 5:13, 6:14, 7:15, 8:16, 9:17, 10:18, 11:19,  12:2, 13:20, 14:21, 15:22, 16:23, 17:24, 18:25, 19:26,
        20:27, 21:28, 22:29, 23:3, 24:30, 25:31, 26:32, 27:33, 28:34, 29:35, 29:35, 30:36, 31:37, 32:38, 33:4, 34:5, 35:6, 36:7, 37:8, 38:9}
    
    if use_batch:
        subimages = np.array(subimages)
        out_batch = []
        for batch_id, subimage in enumerate(subimages):
#             try:
            image_tensor = test_transforms(subimage).float()
#             except:
#                 print(subimage.shape)
            image_tensor = torch.unsqueeze(image_tensor, 0).float()
            out_batch.append(image_tensor[0])
        out_batch = torch.stack(out_batch, axis = 0)
        out_batch = out_batch.to(device)
        output = model(out_batch)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().numpy()
        for pred_id, pred in enumerate(preds):
            preds[pred_id] = class_mapping[pred]
        return preds
    else:
        with torch.no_grad():
            image_tensor = test_transforms(subimages).float()
            image_tensor = torch.unsqueeze(image_tensor, 0).float()
            input = Variable(image_tensor).to(device)
            output = model(input)
            _, preds = torch.max(output, 1)
        pre_cls = preds.cpu().numpy()[0]
        return class_mapping[pre_cls]

def count_klt_image(image_path, rack_json_path, klts_json_path, device):

    exist_klt = []
    image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)

    with open(klts_json_path, 'r') as f:
        labels = json.load(f)
    
    with open(rack_json_path, 'r') as f:
        rack2 = json.load(f)
    
    sub_images = []
    sub_labels = []

    for label_id, label in enumerate(labels):
        sub_image, large_image = get_subimage(image, label, rack2)
        sub_images.append(sub_image)
        sub_labels.append(label)

    start_p = time()
    pos_ids = get_class(sub_images, model, device)
    print(f'Run time: {time() - start_p}, {len(sub_images)}')
    for pi, pos_id in enumerate(pos_ids):
        exist_klt.append([pos_id, sub_labels[pi]])
    
    imshow(get_subimages(image, exist_klt))
    print(np.array(exist_klt, dtype = object)[:, 0])
    return np.array(exist_klt, dtype = object)[:, 0]

# +
def count_klt_image_v2(image, rack_pos, klt_pos, device, model):

    exist_klt = [] 
    sub_images = []
    sub_labels = []

    for label_id, label in enumerate(klt_pos):
        sub_image, large_image = get_subimage_v2(image, label, rack_pos)
        sub_images.append(sub_image)
        sub_labels.append(label)

    start_p = time()
    pos_ids = get_class(sub_images, model, device)
#     print(f'Run time: {time() - start_p}, {len(sub_images)}')
    for pi, pos_id in enumerate(pos_ids):
        exist_klt.append([pos_id, sub_labels[pi]])
    
#     print(np.array(exist_klt, dtype = object)[:, 0])
#     print('\n')
    return np.array(exist_klt, dtype = object)[:, 0]


# -

if __name__ == "__main__":
    # IMAGE_PATH = 'Data/Eval/images/'
    # JSON_PATH = 'Data/Eval/labels/json/'
    # SAVE_PATH = 'Data/DrawVideo/from_train/'
    # SAMPLING_FREQ = 1

    ## Load classifier
    with torch.cuda.device(7):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        model = timm.create_model('tf_efficientnetv2_b2', pretrained=False)
        model.reset_classifier(39)
        model.load_state_dict(torch.load('PositionClassify/EffectionNet_Classifiy/saved_model/best_acc_6078.pth', map_location=torch.device(device)))
        model.eval()
        model = model.to(device)

        result = count_klt_image('./DummyData/100.jpg', 'DummyData/100_rack_2.json', 'DummyData/f100_klts.json', device)
        result.sort()
        print()



        
