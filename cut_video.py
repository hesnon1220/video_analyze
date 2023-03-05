import cv2
import os
from tqdm import tqdm
import time
import threading
import numpy as np
from scipy import stats
def task(video_name,var_hist_path,video_folder_path,save_path):

    fps = 23.98
    min_sec_set = 1
    max_sec_set = 5
    min_interval_set = fps*min_sec_set
    max_interval_set = fps*max_sec_set
    #interval_set = fps*sec_set

    var_hist = []
    with open(os.path.join(var_hist_path,"%s_hist_var.txt"%(video_name)),"r") as txtfile :
        lines = txtfile.readlines()
        for i in lines :
            var_hist.append(eval(i))

                    
    cut_point = []
    rec_state = False
    tmp_hist = []
    for i in range(len(var_hist)) :
        if rec_state : tmp_hist.append(var_hist[i])
        if not rec_state and not ( var_hist[i] < 0.8 ) :
            rec_state = True
            start_frame = i
            continue
        if rec_state and ( var_hist[i] < 0.8 ) :
            rec_state = False
            end_frame = i
            if (( end_frame - 5 ) - ( start_frame + 5 ) > min_interval_set):
                #if not ( np.mix(tmp_hist[5:-5]) > 0.9 ) :
                middle_point = (start_frame + end_frame)/2
                left_point = max(start_frame+5,int( middle_point - max_interval_set / 2 ))
                right_point = min(end_frame-5,int( middle_point + max_interval_set / 2 ))
                if len(var_hist[left_point:right_point]) != 0 :
                    if not ( np.mean(np.array(var_hist[left_point:right_point]) >0.95) > 0.7 ) :
                    #mode = stats.mode(np.array(tmp_hist[5:-5]),axis=None, keepdims=True)[0]
                    #if not mode[0] > 0.9  :
                        cut_point.append( (left_point,right_point) )
                        #cut_point.append( (start_frame+5,end_frame-5) )
            tmp_hist = []

    vidCap = cv2.VideoCapture(os.path.join(video_folder_path,"%s.mp4"%(video_name)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width  = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = vidCap.get(cv2.CAP_PROP_FPS)
    #FPS = 25

    cut_point_idx = 0
    #out = cv2.VideoWriter(os.path.join(cut_video_path,"%s_%s.mp4"%(pic_name,)),fourcc, FPS, (int(frame_width),int(frame_height)))
    frame_num = 0
    
    cut_state = False
    while True :
        ret = vidCap.grab()
        if not ret or cut_point_idx == len(cut_point): break
        if frame_num >=  cut_point[cut_point_idx][0] :
            start_time = time.time()
            if not cut_state :
                cut_state = True
                out = cv2.VideoWriter(os.path.join(save_path,"%s_%s.mp4"%(video_name,cut_point_idx)),fourcc, FPS, (frame_width,frame_height) )
            ret,image = vidCap.retrieve()
            out.write(image)
            if frame_num == cut_point[cut_point_idx][1] and cut_state :
                cut_state = False
                out.release()
                cut_point_idx += 1
            end_time = time.time()
            print(" {} process : {} --- {:.2f} it/s".format(cut_point_idx,frame_num,1/(end_time-start_time)),end="\r")
        frame_num += 1
        

    vidCap.release()
    cv2.destroyAllWindows()

class MyThread(threading.Thread):
    def __init__(self, video_name,var_hist_path,video_folder_path,save_path,semaphore):
        threading.Thread.__init__(self)
        self.video_name = video_name
        self.var_hist_path = var_hist_path
        self.video_folder_path =video_folder_path
        self.save_path = save_path
        self.semaphore = semaphore
    ####################################################################################################################################
    def run(self):
        with self.semaphore :
            task(self.video_name,self.var_hist_path,self.video_folder_path,self.save_path)



def main() :

    var_hist_path = r"F:\work\video_analyze\output\var_hist\Beelzebub-jou no Okinimesu mama"
    video_folder_path = r"F:\work\video_analyze\data\video\Beelzebub-jou no Okinimesu mama"
    save_path = r"F:\work\video_analyze\output\cut_video\Beelzebub-jou no Okinimesu mama"

    max_deals = 10
    semaphore = threading.BoundedSemaphore(max_deals)
    threads = []
    for i in os.listdir(var_hist_path) :
        video_name = i.replace( "_hist_var.txt","" )
        threads.append(MyThread(video_name,var_hist_path,video_folder_path,save_path,semaphore))
    ####################################################################################################################################
    for _idx_ in range(len(threads)) :
        threads[_idx_].start()
    ####################################################################################################################################
    for _idx_ in range(len(threads)):
        threads[_idx_].join()


if __name__ == "__main__" :
    main()