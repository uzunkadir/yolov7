import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def isin(helmet, bbox):
    x1, y1, x2, y2 = bbox
    return (x1 <= helmet.middle_x) and (helmet.middle_x <= x2) and (y1 <= helmet.middle_y) and (helmet.middle_y <= y2)

def dist(helmet, bbox):
    x1, y1, x2, y2 = bbox
    top_middle = [(x1+x2)/2, (y1+y2)/4]
    return ((helmet.middle_x-top_middle[0]) ** 2+ (helmet.middle_y-top_middle[1]) ** 2) ** 0.5

def detect(opt, model,return_img=False):
    
    helmets = opt["filtered_df"]
    stride = opt["stride"]
    imgsz = opt["img_size"]

    set_logging()
    device = select_device(opt["device"])

    # Set Dataloader
    dataset = LoadImages(opt["source"], img_size=imgsz, stride=stride)


    frame_count = 0
    for path, img, im0, vid_cap in dataset:
        frame_count += 1
        
        if frame_count not in opt["frame_set"]:
            continue
        

        helmets_framed = helmets[helmets.frame==frame_count].copy()
        img = torch.from_numpy(img).to(device)
        img = img.half()   # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        with torch.no_grad():   
            pred = model(img, augment=opt["augment"])[0]

        pred = non_max_suppression(pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"], agnostic=opt["agnostic_nms"])
        
        all_coords = []
        for det in pred:  
            
            if len(det):
                
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                                        
                for *xyxy, conf, cls in reversed(det):
                    label_player = ""
                    coords_indv = [int(j) for j in xyxy]
                    all_coords.append(coords_indv)
                    
                            
        for helmet in helmets_framed.itertuples():
            d = [dist(helmet, bbox) if isin(helmet, bbox) else float("inf") for bbox in all_coords]
            if not np.all(np.array(d) == float("inf")):
                d_i = np.argmin(d)
                helmets.loc[helmet.Index,["yolo_x1","yolo_y1","yolo_x2","yolo_y2"]] = np.array(all_coords[d_i])
    
                if opt["bbox"]:
                    label_player = helmets_framed.loc[helmet.Index,"player_label"]
                    plot_one_box(all_coords[d_i], im0, label=label_player, color=(0,0,0), line_thickness=1)
        
        
    

        if opt["bbox"]:
            
            for k in helmets_framed[["left","width","top","height"]].values:
                left,width,top,height = k
                x1,y1,x2,y2 = int(left),int(top),int(left+width),int(top+height)
                im0 = cv2.rectangle(im0, (x1,y1),(x2,y2), (255,0,0), 2)
            
        if opt["show"]:
            cv2.imshow("img_pred", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

    return helmets
        
        
        
        
