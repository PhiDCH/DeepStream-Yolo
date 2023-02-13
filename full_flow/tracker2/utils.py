import cv2
import numpy as np

class_dict = {"rack_1":0, "rack_2":1, "rack_3":2, "rack_4":3, "klt_box_empty":4, "klt_box_full":5}
class_dict_inv = {v: k for k, v in class_dict.items()}
color_dict = [(255,0,0), (200,0,50), (150, 0, 100), (100, 0, 150), (50, 0, 200), (0, 0, 250)]
img_h, img_w = 720, 1280

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlbrs, obj_ids, cl_ids, color=None, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 1.2
    text_thickness = 2
    line_thickness = 2

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlbrs)),
                (0, int(20 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=1)

    for i, tlbr in enumerate(tlbrs):
        x1, y1, x2, y2 = tlbr
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        obj_id = int(obj_ids[i])
        obj_cl = int(cl_ids[i])
        obj_name = class_dict_inv[obj_cl]
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        if color is None:
            color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def unpack(lb):
    out = []
    for obj in lb:
        cl_id = class_dict[obj["ObjectClassName"]]
        x1,y1,x2,y2 = obj['Left'], obj['Top'], obj['Right'], obj['Bottom']
        out.append([cl_id, x1, y1, x2, y2, 1])
    return np.array(out, dtype=np.float)

def draw(img, lbs):
    fontscale = 0.7
    for lb in lbs:
        conf = lb[-1]
        if conf <0.3:
            continue
        cl = lb[0]
        # if cl >= 4:
        #     continue
        name = f"{conf:.2f}" 
        # name = f"{conf:.2f}"
        color = color_dict[int(cl)]
        x,y,w,h = lb[1:5]
        box = int((x-w/2)*img_w), int((y-h/2)*img_h), int((x+w/2)*img_w), int((y+h/2)*img_h) 
        cv2.rectangle(img, box[:2], box[2:], color, 2, cv2.LINE_AA)
        txt_color = (255, 255, 255)
        w, h = cv2.getTextSize(name, 0, fontScale=fontscale, thickness=2)[0]
        p2 = box[0] + w, box[1] - h - 5
        cv2.rectangle(img, box[:2], p2, color, -1, cv2.LINE_AA)  # filled

        cv2.putText(img, name, (box[0], max(h + 3, box[1] - 5)), 0, fontscale, txt_color,
                    thickness=1, lineType=cv2.LINE_AA)
    return img