import librosa
import numpy as np
from Helper_private import get_fft
import cv2
import os
import matplotlib.pyplot as plt

def min_index(lst):
    min_val = min(lst)
    return [i for i, x in enumerate(lst) if x == min_val]



def main(video_name,var_hist_path,video_folder_path,save_path):
    fps = 23.976
    
    #audio_path = os.path.join(r"F:\work\video_analyze\separated\htdemucs\test",i)
    audio_path = r"F:\work\video_analyze\data\audio\Detective Conan The Culprit Hanzawa\01.捕まえて、今夜。.flac"
    y,sr = librosa.load(audio_path)
    fft_data = get_fft(audio_path,fps)
    print(np.array(fft_data).shape)
    
    totla_frame = int(len(y)/sr*fps)

    ##########################################################################################################################
    vocal_path = r"F:\work\video_analyze\data\audio\Detective Conan The Culprit Hanzawa\separated\htdemucs\01.捕まえて、今夜。\vocals.wav"
    vocal_frams = get_point(vocal_path,fps)


    print(vocal_frams)
    ##########################################################################################################################
    
    #print(y.shape)
    #print(sr)
    with open(r"F:\work\video_analyze\output\lnc_time.txt","r") as txt_file :
        line = txt_file.readline()
        lnc_time = np.array(list(map(float,line.replace("\r","").split("\t"))))
    vocal_cut = np.array((lnc_time-0.3)*fps)
    print(vocal_cut)


    tmp_vocal_frams = np.array([ i[0] for i in vocal_frams])
    for idx,ele in enumerate(vocal_cut) :
        dist = np.abs(tmp_vocal_frams-ele)
        if min(dist) < 3*fps :
            vocal_cut[idx] = tmp_vocal_frams[min_index(dist)[0]]

    vocal_cut = np.array(vocal_cut,dtype="uint64")
    print(vocal_frams)
    print(vocal_cut)

    
    interlude = []
    start_inter = 0
    for i in vocal_frams :
        end_inter = i[0]
        if end_inter != start_inter :
            interlude.append( [start_inter,end_inter] )
        start_inter = i[1]
    if start_inter != totla_frame :
        interlude.append( [start_inter,totla_frame] )
    print(interlude)
    

    paragraph_dict = {
        "interlude" : [],
        "vocal" : {}
    }

    for i in interlude :
        paragraph_dict["interlude"].append( i[1]-i[0] )
    vocal_frame_idx = 0

    for vocal_frame_idx in range(len(vocal_frams)) :
        tmp = [vocal_frams[vocal_frame_idx][0]] + [ i for i in vocal_cut if (i > vocal_frams[vocal_frame_idx][0] and i < vocal_frams[vocal_frame_idx][1]) ] + [vocal_frams[vocal_frame_idx][1]]
        paragraph_dict["vocal"][vocal_frame_idx] = []
        for i in range(len(tmp)-1) :
            paragraph_dict["vocal"][vocal_frame_idx] .append( tmp[i+1] - tmp[i] )

    print(paragraph_dict)



def get_cut_video(var_hist_path,video_name) :
    
    ##########################################################################################################################
    fps = 23.976
    start_point = int(fps*(1*60+40))
    end_point = int(fps*(9*60+10))

    min_sec_set = 1
    min_interval_set = fps*min_sec_set
    #interval_set = fps*sec_set

    var_hist = []
    with open(os.path.join(var_hist_path,"%s_hist_var.txt"%(video_name)),"r") as txtfile :
        lines = txtfile.readlines()
        for i in lines :
            var_hist.append(eval(i))

                    
    cut_point_dict = {}
    cut_point_idx = 0
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
            left_point = start_frame+5
            right_point = end_frame-5
            if (right_point -left_point ) > min_interval_set:
                if not ( np.mean(np.array(var_hist[left_point:right_point]) >0.95) > 0.7 ) :
                    cut_point_dict[cut_point_idx] = {
                        "interval" : (left_point,right_point) ,
                        "lenght" : right_point-left_point
                    }
                    cut_point_idx += 1
            tmp_hist = []

    return cut_point_dict


def get_point(audio_file,fps):
    # 讀取音訊檔案
    y_audio, sr_audio = librosa.load(audio_file)

    # 設定窗口大小和跨度
    frame_length = 64
    hop_length = 512

    # 計算短時能量
    energy = librosa.feature.rms(y=y_audio, frame_length=frame_length, hop_length=hop_length)[0]
    ZCR = librosa.feature.zero_crossing_rate(y=y_audio, frame_length=frame_length, hop_length=hop_length)[0]

    energy_x = np.arange(0,len(energy),1)
    energy_frame = np.array(energy_x*hop_length/sr_audio*fps,dtype = "uint64")
    energy_gap = np.array(energy > 0.1)
    energy_gap_con = np.zeros(len(energy_gap))
    start_idx = 0
    end_idx = 0
    while start_idx < (len(energy_gap)-1) :
        if energy_gap[start_idx] == 1 :
            tmp_gap = energy_gap[start_idx+1:min(len(energy_gap),int(start_idx+5*fps))]
            if max(tmp_gap) == 1 :
                for tmp_idx in range( len(tmp_gap)-1,-1 ,-1) :
                    if tmp_gap[tmp_idx] == 1 :
                        end_idx = ( start_idx + 1 ) + tmp_idx
                        energy_gap_con[start_idx:end_idx] = 1
                        start_idx = end_idx - 1
                        break
        start_idx += 1

    if_rec = False
    vocal_frams = []
    for idx,ele in enumerate(energy_gap_con) :
        if ele == 1 and not if_rec :
            if_rec = True
            start_frame = energy_frame[idx]
        elif ( ele == 0 or idx == len(energy_gap_con)-1 ) and if_rec :
            if_rec = False
            end_frame =  energy_frame[idx]
            vocal_frams.append( [start_frame,end_frame] )

    """
    x_bar = np.arange( 0 , len( energy ) , 1 )
    fig = plt.figure(figsize=(70,8))
    plt.subplot(111)
    plt.plot( energy_frame , ZCR , c = "b" )
    plt.plot( energy_frame , energy , c = "r" )
    plt.plot( energy_frame , np.array(energy > 0.1) , c = "g" )
    plt.plot( energy_frame , energy_gap_con, c = "purple")
    #plt.scatter(tmp_data/fps, np.array(cut_result)[tmp_data])
    plt.savefig(os.path.join(r"F:\work\video_analyze\output","{}.png".format("cut_reslut_2")),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
    """

    # 尋找開始發聲的位置
    onset_frames = librosa.onset.onset_detect(y=y_audio, sr=sr_audio, hop_length=hop_length, backtrack=True, energy=ZCR, 
                                            units='frames', pre_max=1, post_max=1, pre_avg=1, post_avg=1, 
                                            delta=0.15, wait=10)

    # 將帧位置轉換為秒
    onset_times = librosa.frames_to_time(onset_frames, sr=sr_audio, hop_length=hop_length)


    return vocal_frams

if __name__ == "__main__" :
    video_name = ""
    var_hist_path = r"F:\work\video_analyze\output\var_hist\Detective Conan The Culprit Hanzawa"
    video_folder_path = r""
    save_path = r""
    #main(video_name,var_hist_path,video_folder_path,save_path)
    cut_point_dict = get_cut_video(var_hist_path,r"[GST] Detective Conan The Culprit Hanazawa - S01E01 [1080p]")