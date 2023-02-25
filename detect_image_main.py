import os
import sys
sys.path.append("E:/GITHUB-REPOS/NFL-Contact-Detection")
import cv2
from detect_image import detect
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import  TracedModel
from utils.torch_utils import select_device
from torch import cuda
from datetime import datetime 
import pandas as pd
from tqdm import tqdm
print(cuda.is_available())



# WKDIR            = '/kaggle/input/nfl-player-contact-detection'
WKDIR            = 'csvs'

FRAME_PER_SECOND = 59.94
SECOND_PER_FRAME = 1 / FRAME_PER_SECOND


PARAMS_METADATA = {
    "usecols"    : ["game_play", "view", "start_time","snap_time"], 
    "parse_dates": ["start_time","snap_time"]
    }


PARAMS_HELMETS = {
    "usecols"  : ["game_play","nfl_player_id","view","frame","left","width","top","height","player_label"],
    "dtype"    : {"nfl_player_id":"str"}
    }


############ TRAIN #############################
metadata = pd.read_csv(r"E:\GITHUB-REPOS\NFL-Contact-Detection\csvs\train_video_metadata.csv", **PARAMS_METADATA)
helmets  = pd.read_csv(r"E:\GITHUB-REPOS\NFL-Contact-Detection\csvs\train_baseline_helmets.csv", **PARAMS_HELMETS)
################################################

metadata.drop("snap_time", axis=1, inplace=True)


metadata = metadata.set_index(["game_play","view"])
helmets  = helmets.set_index(["game_play","view"])


metadataJoined = metadata.join(helmets,on=("game_play", "view"), how="inner")

metadataJoined["datetime"        ] = pd.to_timedelta(metadataJoined.frame * SECOND_PER_FRAME, unit="seconds") + metadataJoined.start_time
metadataJoined["datetime_rounded"] = metadataJoined.datetime.dt.round(freq="0.1S")
metadataJoined["datetime_diff"   ] = abs(metadataJoined.datetime_rounded - metadataJoined.datetime)

metadataJoined.set_index(["nfl_player_id", "datetime_rounded", "datetime_diff"], append=True, inplace=True)
metadataJoined.drop(["start_time","datetime"], axis=1, inplace=True)
metadataJoined.index.set_names("datetime", level="datetime_rounded", inplace=True)

metadataJoined.sort_index(level=["game_play","view","datetime","datetime_diff"], inplace=True)

metadajoinined_first = metadataJoined.groupby(by=["game_play","nfl_player_id","datetime","view"]).first()
metadajoinined_first.reset_index(inplace=True)

metadajoinined_first = metadajoinined_first.sort_values(["game_play","view","datetime","frame"])

date_frame = metadataJoined.groupby(by=["game_play","datetime","view"]).first()
date_frame.reset_index(inplace=True)
date_frame = date_frame[["game_play","datetime","view","frame"]]



helmets_time_rel = metadajoinined_first.merge(
                                            date_frame,
                                            how="right",
                                            left_on=("game_play","datetime","view","frame"),
                                            right_on=("game_play","datetime","view","frame")
                                            )
                                            
    
# helmets_time_rel = metadajoinined_first.copy()
helmets_time_rel = helmets_time_rel.assign(middle_x =((helmets_time_rel["left"]+helmets_time_rel["left"]+helmets_time_rel["width"])/2).values) 
helmets_time_rel = helmets_time_rel.assign(middle_y =((helmets_time_rel["top"]+helmets_time_rel["top"]+helmets_time_rel["height"])/2).values) 



device_type = "gpu"
model_name = r"E:\GITHUB-REPOS\NFL-Contact-Detection\yolov7\yolo_models\yolov7-e6e.pt"
device = select_device(device_type)
model = attempt_load(model_name, map_location=device)  
model.half()


height, width = 720, 1280


for i in tqdm(helmets_time_rel.groupby(["game_play","view"])):
    
    filtered_df = i[1]
    video_name = "_".join(i[0])
    frame_set = set(filtered_df.frame.values)
    
    opt = {'stride':int(model.stride.max()),
            'source': f"E:/GITHUB-REPOS/NFL-Contact-Detection/train/{video_name}.mp4",
            'img_size': width,
            "model_name":model_name,
            'conf_thres': 0.05,
            'iou_thres': 0.45,
            'device': device_type,
            'classes': (0,),
            'agnostic_nms': False,
            'augment': False,
            'show': 0,
            'bbox': 0,
            "frame_set":frame_set,
            "filtered_df":filtered_df}
    
    helmets_yolo = detect(opt, model, return_img=True)
    helmets_time_rel.loc[helmets_yolo.index, ["yolo_x1","yolo_y1","yolo_x2","yolo_y2"]] = helmets_yolo[["yolo_x1","yolo_y1","yolo_x2","yolo_y2"]].values


# helmets_time_rel = helmets_time_rel.fillna(0)
# helmets_time_rel.to_csv(r"E:\GITHUB-REPOS\NFL-Contact-Detection\csvs\helmets_timerel_yolo.csv")

# helmets_time_rel[(helmets_time_rel.yolo_x1==0)&
#                  (helmets_time_rel.yolo_y1==0)&
#                  (helmets_time_rel.yolo_x2==0)&
#                  (helmets_time_rel.yolo_y2==0)]


