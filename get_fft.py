import numpy as np
from scipy.io import wavfile
import cv2
from tqdm import tqdm
import os
from moviepy.editor import AudioFileClip,VideoFileClip
import argparse
from pathlib import Path
import time
import json
from Helper_private import *
import scipy.signal
import threading
import yaml

video_path = r"F:\work\video_analyze\my_work\video"
sound_path = r"F:\work\video_analyze\my_work\sound\separated\htdemucs"
sound_class = ["bass","drums","vocals","other"]
point_path = r"F:\work\video_analyze\my_work\video\data\point"
output_path = r"F:\work\video_analyze\my_work\train_data"


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

    cut_result = []
    for i in result :
        cut_result.append(np.max(i))

    #draw_single_lines(np.array(cut_result),r"F:\work\video_analyze\output","cut_reslut")

    #peaks = scipy.signal.find_peaks_cwt(cut_result,5)

    return cut_result


def task(work_file):


    if ".mp4" not in work_file : return
    work_name = work_file.replace(".mp4","")


    vidCap = cv2.VideoCapture(os.path.join( video_path , work_file ))
    fps = vidCap.get(cv2.CAP_PROP_FPS)

    point_data = []
    with open(os.path.join(point_path,"{}_point.txt".format(work_name)),"r") as txtfile :
        lines = txtfile.readlines()
        for i in lines :
            point_data.append(eval(i))

    sound_data = {
        "point_data" : point_data
    }

    for sc in sound_class :
        sound_file_path = os.path.join(sound_path,work_name,"{}.wav".format(sc)  )
        sound_data[sc] = get_fft_max(file_path=sound_file_path,fps=fps)


    with open(os.path.join(output_path,"{}.yml".format(work_name)), 'w') as f:
        yaml.dump(sound_data, f)
    


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