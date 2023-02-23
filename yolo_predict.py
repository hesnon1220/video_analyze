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
    #predict_model.iou = 0.3
    #predict_model.conf = 0.5
    predict_model.to(device)


    with open( "coco128.yml" ,"r" ) as yamlfile :
        class_name = yaml.load(yamlfile,Loader=yaml.Loader)

    print(class_name["names"])
    

    vidCap = cv2.VideoCapture(r"F:\work\video_analyze\output\cut_video\Beelzebub-jou no Okinimesu mama\[Erai-raws] Beelzebub-jou no Okinimesu mama - 01 [720p][Multiple Subtitle]_11.mp4")

    while True :
        ret = vidCap.grab()
        if not ret : break
        ret,image = vidCap.retrieve()
        predict_result = predict_model(image)
        data_frame = dataframe_change(predict_result.pandas().xyxy)[0]
        item_list = []
        for i in data_frame :
            item_list.append(class_name["names"][i[-1]])
        print(item_list)
    




if __name__ == "__main__" :
    main()