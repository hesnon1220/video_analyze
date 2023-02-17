import cv2
import numpy as np
from Helper import *
from tqdm import tqdm
import time

data_floder = r"data"
output_floder = r"output"



def main():
    video_list = os.listdir(data_floder)
    print(video_list)
    

    video_path = os.path.join(data_floder,video_list[1])
    vidCap = cv2.VideoCapture(video_path)
    fps = 10
    loop_num = 10
    frame_num = 0
    stop_num = 0

    hist_list = []


    while True :
        ret = vidCap.grab()
        if not ret : break
        #if stop_num == 36000 : break
        if frame_num == 0 :
            start_time = time.time()
            ret,image = vidCap.retrieve()
            #tmp_img = cv2.resize(image,[256,256])
            hist_list.append(get_hist(image))
            stop_num += 1
            end_time = time.time()
            print("process : {} --- {:.2f} it/s".format(stop_num,1/(end_time-start_time)),end="\r")
        frame_num = ( frame_num + 1 ) % loop_num
        
    

    #hist_var = []
    #for i in range(len(hist_list)-1) :
        #hist_var.append(hist_similar(hist_list[i],hist_list[i+1]))
    #print(hist_var)

    with open(os.path.join(output_floder,"hist_var.txt"),"w") as txtfile:
        for i in range(len(hist_list)-1) :
            print(hist_similar(hist_list[i],hist_list[i+1]),file=txtfile)
        




if __name__ == "__main__" :
    main()

