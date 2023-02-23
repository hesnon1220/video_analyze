from typing import List
from Helper_private import coord_change,dataframe_change,sigmoid,fixed_area_detect
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np
import time
import yaml



def main() :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predict_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
    predict_model.iou = 0.3
    predict_model.conf = 0.5

    predict_model.to(device)

    img = cv2.imread(r"D:\video_analyze\data\n-b8c7ab59772e092f,ch1_s-20220920180000_e-190000.mp4_20230220_101007.380.jpg")
    predict_result = predict_model(img)
    data_frame = dataframe_change(predict_result.pandas().xyxy)
    print(data_frame)
    pass


if __name__ == "__main__" :
    main()