from typing import List
from Helper_private import coord_change,dataframe_change,sigmoid,fixed_area_detect,shorten_number
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np
import time
import yaml
import threading




def video_predict(video_path,predict_model) :
    class_num = ["black","text","title"]
    vidCap = cv2.VideoCapture(video_path)
    return_dict = {
        "frame_num" : 0 ,
        "black" : 0,
        "text" : 0,
        "title" : 0,
        "gray_mean" : [],
        "gray_std" : [],
    }
    while True :
        ret = vidCap.grab()
        if not ret : break
        ret,image = vidCap.retrieve()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return_dict["gray_mean"].append( shorten_number(np.mean(gray_image)))
        return_dict["gray_std"].append(shorten_number(np.std(gray_image)))
        predict_result = predict_model(image)
        data_frame = dataframe_change(predict_result.pandas().xyxy)[0]
        for i in data_frame :
            if i[-1] == 0 :
                if i[-2] >= 0.9 : return_dict["black"] += 1
            else : return_dict[class_num[int(i[-1])]] += 1
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
    predict_model = torch.hub.load('ultralytics/yolov5', 'custom', path = r"F:\work\yolov5\runs\train\exp9\weights\best.pt")
    predict_model.iou = 0.2
    predict_model.conf = 0.2
    predict_model.to(device)

    base_path = r"F:\work\video_analyze\output\cut_video\Detective Conan The Culprit Hanzawa"
    for video_name in os.listdir(base_path) :
        threads.append(MyThread(video_dict,base_path,video_name,predict_model,semaphore))
    ####################################################################################################################################
    for _idx_ in range(len(threads)) :
        threads[_idx_].start()
    ####################################################################################################################################
    for _idx_ in range(len(threads)):
        threads[_idx_].join()

    with open('Hanzawa_video_dict.yaml', 'w') as f:
        yaml.dump(video_dict, f)


if __name__ == "__main__" :
    main()