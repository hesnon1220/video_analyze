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

def main():


    file_path = r"F:\work\video_analyze\separated\htdemucs\test\drums.wav"

    fps = 30.0
    bar_num = 150

    sampling_interval = 1/fps   #間隔

    root, extension = os.path.splitext(file_path)    #獲取檔案副檔名
    print(root,extension)
    if extension != ".wav" :    #轉檔wav
        file_path = mp32wav(root,extension)
    samplerate, data = wavfile.read(file_path)   #獲取採樣率、振幅(雙聲道)

    result = fft(data,samplerate,sampling_interval,bar_num) #傅立葉轉換

    cut_result = []
    for i in result :
        cut_result.append(np.max(i[:5]))

    draw_single_lines(np.array(cut_result),r"F:\work\video_analyze\output","cut_reslut")
    
    #peaks = scipy.signal.find_peaks_cwt(cut_result,5)

    
    
    pass


if __name__ == "__main__" :
    main()