from typing import List
from Helper_private import coord_change,dataframe_change,sigmoid,fixed_area_detect
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np
import time
import yaml
import threading




def video_predict(video_path,predict_model) :
    class_num = ["charater","creature","text"]
    vidCap = cv2.VideoCapture(video_path)
    return_dict = {
        "frame_num" : 0 ,
        "charater" : 0 ,
        "creature" : 0 ,
        "text" : 0
    }
    fram_num = 0
    while True :
        ret = vidCap.grab()
        if not ret : break
        ret,image = vidCap.retrieve()
        predict_result = predict_model(image)
        data_frame = dataframe_change(predict_result.pandas().xyxy)[0]
        for i in data_frame :
            return_dict[class_num[int(i[-1])]] += 1
        return_dict["frame_num"] += 1
    vidCap.release()
    cv2.destroyAllWindows()

    return return_dict

def task(video_dict,base_path,video_name,predict_model) :
    start_time = time.time()
    video_path = os.path.join( base_path , video_name )
    video_name_nomp4 = video_name.replace(".mp4","")
    result_dict = video_predict(video_path,predict_model)
    video_dict[video_name_nomp4] = result_dict
    end_time = time.time()
    print("{} --- {:.2f} it/s.".format(video_name,1/(end_time-start_time)))

class MyThread(threading.Thread):
    def __init__(self, video_dict,base_path,video_name,predict_model,semaphore):
        threading.Thread.__init__(self)
        self.video_dict = video_dict
        self.base_path = base_path
        self.video_name = video_name
        self.predict_model =predict_model
        self.semaphore = semaphore
    ####################################################################################################################################
    def run(self):
        with self.semaphore :
            task(self.video_dict,self.base_path,self.video_name,self.predict_model)


def main() :
    max_deals = 10
    semaphore = threading.BoundedSemaphore(max_deals)
    threads = []

    video_dict = {}

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    predict_model = torch.hub.load('ultralytics/yolov5', 'custom', path = r"F:\work\yolov5\runs\train\exp3\weights\best.pt")
    predict_model.iou = 0.3
    predict_model.conf = 0.5
    predict_model.to(device)

    base_path = r"F:\work\video_analyze\output\cut_video\Beelzebub-jou no Okinimesu mama"
    for video_name in os.listdir(base_path) :
        threads.append(MyThread(video_dict,base_path,video_name,predict_model,semaphore))
    ####################################################################################################################################
    for _idx_ in range(len(threads)) :
        threads[_idx_].start()
    ####################################################################################################################################
    for _idx_ in range(len(threads)):
        threads[_idx_].join()

    with open('video_dict.yaml', 'w') as f:
        yaml.dump(video_dict, f)


if __name__ == "__main__" :
    main()