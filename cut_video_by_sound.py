import librosa
import numpy as np
from Helper_private import get_fft
import cv2
import os

def main(video_name,var_hist_path,video_folder_path,save_path):
    fps = 23.976
    
    #audio_path = os.path.join(r"F:\work\video_analyze\separated\htdemucs\test",i)
    audio_path = r"F:\work\video_analyze\data\audio\Detective Conan The Culprit Hanzawa\01.捕まえて、今夜。.flac"
    y,sr = librosa.load(audio_path)
    fft_data = get_fft(audio_path,fps)
    print(np.array(fft_data).shape)
    
    #print(y.shape)
    #print(sr)
    with open(r"F:\work\video_analyze\output\lnc_time.txt","r") as txt_file :
        line = txt_file.readline()
        lnc_time = np.array(list(map(float,line.replace("\r","").split("\t"))))
    vocal_cut = np.array((lnc_time-0.3)*fps,dtype = "uint64")
    print(vocal_cut)

    invert_data = []
    step_num = 0
    for i in range(len(vocal_cut)) :
        invert_data.append(int(vocal_cut[i]-step_num))
        step_num = vocal_cut[i]
    invert_data.append(int(len(fft_data)-step_num ))
    print(invert_data)
    ##########################################################################################################################

    start_point = int(fps*(1*60+40))
    end_point = int(fps*(9*60+10))



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
    for i in range(start_point,end_point) :
        if rec_state : tmp_hist.append(var_hist[i])
        if not rec_state and not ( var_hist[i] < 0.8 ) :
            rec_state = True
            start_frame = i
            continue
        if rec_state and ( ( var_hist[i] < 0.8 ) or i == end_point-1) :
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







if __name__ == "__main__" :
    main()