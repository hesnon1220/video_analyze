import cv2
import numpy as np
from Helper_private import *
from tqdm import tqdm
import time
import threading

def task(video_name,data_floder,output_floder) :
    video_path = os.path.join(data_floder,video_name)
    video_name = video_name.replace(".mp4","")
    vidCap = cv2.VideoCapture(video_path)
    #fps = 23
    loop_num = 1
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
            print(" {} process : {} --- {:.2f} it/s".format(video_name,stop_num,1/(end_time-start_time)),end="\r")
        frame_num = ( frame_num + 1 ) % loop_num

    #hist_var = []
    #for i in range(len(hist_list)-1) :
        #hist_var.append(hist_similar(hist_list[i],hist_list[i+1]))
    #print(hist_var)

    with open(os.path.join(output_floder,"{}_hist_var.txt".format(video_name)),"w") as txtfile:
        for i in range(len(hist_list)-1) :
            print(hist_similar(hist_list[i],hist_list[i+1]),file=txtfile)
    vidCap.release()
    cv2.destroyAllWindows()

class MyThread(threading.Thread):
    def __init__(self, video_name,data_floder,output_floder,semaphore):
        threading.Thread.__init__(self)
        self.video_name = video_name
        self.data_floder = data_floder
        self.output_floder = output_floder
        self.semaphore = semaphore
    ####################################################################################################################################
    def run(self):
        with self.semaphore :
            task(self.video_name,self.data_floder,self.output_floder)
def main():

    data_floder = r"F:\work\video_analyze\data\video\Beelzebub-jou no Okinimesu mama"
    output_floder = r"F:\work\video_analyze\output\var_hist\Beelzebub-jou no Okinimesu mama"

    max_deals = 16
    semaphore = threading.BoundedSemaphore(max_deals)
    threads = []
    for i in os.listdir(data_floder) :
        threads.append(MyThread(i,data_floder,output_floder,semaphore))
    ####################################################################################################################################
    for _idx_ in range(len(threads)) :
        threads[_idx_].start()
    ####################################################################################################################################
    for _idx_ in range(len(threads)):
        threads[_idx_].join()



if __name__ == "__main__" :
    main()

