import numpy as np
from scipy.io import wavfile
import cv2
from tqdm import tqdm
import os
from moviepy import AudioFileClip,VideoFileClip
import argparse
from pathlib import Path
import time
import json
from Helper_private import *
import scipy.signal
import threading
import yaml
from sklearn.preprocessing import StandardScaler

video_path = r"F:\work\video_analyze\my_work\video"
sound_path = r"F:\work\video_analyze\my_work\sound"
#sound_class = ["bass","drums","vocals","other"]
point_path = r"F:\work\video_analyze\my_work\video\data\point"
output_path = r"F:\work\video_analyze\my_work\sound\fft_data"
BIES_path = r"F:\work\video_analyze\my_work\sound\BIES_data"



def make_BIES(input_list):
    return_list = []
    len_list = []
    tmp = []
    for idx,ele in enumerate(input_list) :
        tmp.append(ele)
        if ele == 1 :
            if len(tmp) == 1 :
                return_list.append("S")
            else :
                return_list.append("B")
                for _C_ in range( len( tmp )-2 ) :
                    return_list.append("I")
                return_list.append("E")
            len_list.append(len(tmp))
            tmp = []
    for i in range(len(input_list)-len(return_list)) :
        return_list.append("[end]")

    return return_list,len_list


def get_fft_max(file_path,fps):

    sampling_interval = 1/fps   #間隔

    root, extension = os.path.splitext(file_path)    #獲取檔案副檔名
    #print(root,extension)
    if extension != ".wav" :    #轉檔wav
        file_path = mp32wav(root,extension)
    samplerate, data = wavfile.read(file_path)   #獲取採樣率、振幅(雙聲道)
    #print(samplerate)
    #print(data.shape)
    
    result = fft(data,samplerate,sampling_interval) #傅立葉轉換



    #print(np.shape(result))
    #print(result[1000])
    #cut_result = []
    #for i in result :
        #cut_result.append(np.max(i))

    #draw_single_lines(np.array(cut_result),r"F:\work\video_analyze\output","cut_reslut")

    #peaks = scipy.signal.find_peaks_cwt(cut_result,5)

    return result


def task(work_file):


    if ".mp4" not in work_file : return
    work_name = work_file.replace(".mp4","")
    print(work_name)

    vidCap = cv2.VideoCapture(os.path.join( video_path , work_file ))
    fps = vidCap.get(cv2.CAP_PROP_FPS)

    point_data = []
    with open(os.path.join(point_path,"{}_point.txt".format(work_name)),"r") as txtfile :
        lines = txtfile.readlines()
        for i in lines :
            point_data.append(eval(i))



    BIES_data , len_list = make_BIES(point_data)
    point_bies_data = {
        "point_data" : BIES_data,
        "len_data" : len_list
    }

    sound_file_path = os.path.join(sound_path,"{}.wav".format(work_name))
    fft_result = get_fft_max(file_path=sound_file_path,fps=fps)

    #T_array = np.array(fft_result).T
    #scaler = StandardScaler()
    #for i in range(np.shape(T_array)[0]) :
        #scaler = scaler.fit(T_array[i].reshape(-1, 1))
        #T_array[i] = np.array(scaler.transform(T_array[i].reshape(-1, 1))).T
    
    #print(np.shape(T_array.T)) #n時間長度*735向量參數
    #sound_data["fft_data"] = list(T_array.T)
    #for sc in sound_class :
    #    sound_file_path = os.path.join(sound_path,work_name,"{}.wav".format(sc)  )
    #    sound_data[sc] = get_fft_max(file_path=sound_file_path,fps=fps)


    #print(point_data)
    #print(make_BIES(point_data))
    with open(os.path.join(output_path,"{}.yml".format(work_name)), 'w') as f:
        yaml.dump(list(fft_result), f)
    
    with open(os.path.join(BIES_path,"{}.yml".format(work_name)), 'w') as f:
        yaml.dump(point_bies_data, f)

class MyThread(threading.Thread):
    def __init__(self, work_file,semaphore):
        threading.Thread.__init__(self)
        self.work_file = work_file
        self.semaphore = semaphore
    ####################################################################################################################################
    def run(self):
        with self.semaphore :
            task(self.work_file)

def main():

    max_deals = 16
    semaphore = threading.BoundedSemaphore(max_deals)
    threads = []
    for work_file in os.listdir(video_path) :
        threads.append(MyThread(work_file,semaphore))
    ####################################################################################################################################
    for _idx_ in range(len(threads)) :
        threads[_idx_].start()
    ####################################################################################################################################
    for _idx_ in range(len(threads)):
        threads[_idx_].join()



if __name__ == "__main__" :
    main()
    #get_fft_max(r"F:\work\video_analyze\my_work\sound\[乖離性ミリオンアーサー][Million Ways=One Destination][NEET].wav",30)