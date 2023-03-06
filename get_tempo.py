import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
import cv2
from tqdm import tqdm
from moviepy.editor import AudioFileClip,VideoFileClip
from Helper_private import *
import wave
import pylab as pl
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import cluster
import re
import random
import yaml


def chage_speed() :
    pass


def main() :
    
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
    print(np.array(lnc_time*fps,dtype = "uint64"))
    vocal_cut = np.array(lnc_time*fps,dtype = "uint64")
    
    invert_data = []
    step_num = 0
    for i in range(len(vocal_cut)) :
        invert_data.append(int(vocal_cut[i]-step_num))
        step_num = vocal_cut[i]
    invert_data.append(int(len(fft_data)-step_num ))
    print(invert_data)
    

    tempo,beats = librosa.beat.beat_track(y=y,sr=sr)


    #print(tempo,beats)

    #print(librosa.frames_to_time(beats, sr=sr))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr,aggregate=np.median)


    #tmp_beats = []
    #for i in range(len(beats)):
    #    if i%4 == 0 :
    #        tmp_beats.append(beats[i])
    #tmp_time = librosa.frames_to_time(tmp_beats, sr=sr)
    #point_cut = np.array(np.array(tmp_time)*fps,dtype="uint64")


    #fft_data = get_fft(vocal_path,fps)

    
    data_dict = get_cut_video_dict()
    print(data_dict)
    
    sorted_cut_data = sorted(list(map(int,data_dict.keys())),reverse=True)

    print(sorted_cut_data)


    count_dict = {}
    for i in sorted_cut_data :
        count_dict[i] = len(data_dict[i])

    print(count_dict)

    rec_dict = fill_fields(invert_data,count_dict)

    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    #輸出影像設定
    out = cv2.VideoWriter( os.path.join(r"F:\work\video_analyze\test.avi"), fourcc, fps, (1920,1080))   #輸出無聲長條影像

    output_frame = 0
    cut_video_path = r"F:\work\video_analyze\output\cut_video\Detective Conan The Culprit Hanzawa"
    vocal_idx = 0

    space_img = np.zeros(( 1080,1920, 3), np.uint8)

    
    totla_frame = int(len(y)/sr*fps)
    vocal_cut_withend = np.append(vocal_cut,[totla_frame])
    last_image = space_img.copy()
    last_output_frame = 0
    while output_frame < totla_frame :
        last_output_frame = output_frame
        if output_frame >= vocal_cut_withend[vocal_idx] :
            vocal_idx += 1
        if str(vocal_idx) in rec_dict.keys() :
            #for video_len in rec_dict[str(vocal_idx)] :
            if len(rec_dict[str(vocal_idx)]) > 0 :
                video_len = rec_dict[str(vocal_idx)].pop(0)
                print(output_frame,vocal_idx,video_len,data_dict[video_len[0]])
                if len(data_dict[video_len[0]]) != 0 :
                    taked_ele = data_dict[video_len[0]].pop(0) #random.randint(0,len(data_dict[video_len])-1))
                    video_name = "{}_{}.mp4".format(taked_ele[0],taked_ele[1])
                    video_path = os.path.join(cut_video_path,video_name)
                    vidCap = cv2.VideoCapture(video_path)
                    length = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))
                    speed_spot = np.linspace(0,length,length+1) * max(length,video_len[1]) / length
                    img_idx = 0
                    #while tmp_len < video_len[1] :
                    for write_idx in range(video_len[1]) :
                        if write_idx >=  speed_spot[img_idx] :
                            ret = vidCap.grab()
                            if not ret or output_frame >= totla_frame: break
                            ret,image = vidCap.retrieve()
                            #cv2.putText(image, "{}-{}-{}".format(vocal_idx,output_frame,write_idx) , (10, 230), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 0), 1, cv2.LINE_AA)
                            #cv2.putText(image, "{}".format(video_name) , (10, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 1, cv2.LINE_AA)
                            img_idx += 1
                            #last_image = image.copy()
                        out.write(image)
                        output_frame += 1
                    vidCap.release()
                #elif output_frame < vocal_cut_withend[vocal_idx] :
                    #out.write(last_image)
                    #output_frame += 1
        if output_frame == last_output_frame :
            print(output_frame)
            out.write(space_img)
            output_frame += 1
    out.release()   #清理記憶體
    cv2.destroyAllWindows()
    

    #audioclip = AudioFileClip(r"F:\work\video_analyze\data\audio\Beelzebub-jou no Okinimesu Mama\separated\htdemucs\01.ピンクレモネード\vocals.wav") #獲取音頻
    audioclip = AudioFileClip(audio_path)
    clip = VideoFileClip( os.path.join(r"F:\work\video_analyze\test.avi"))    #獲取影片
    new_video = clip.set_audio(audioclip)   #影片合併音頻
    new_video.write_videofile( os.path.join(r"F:\work\video_analyze\test_1.mp4")) 
    

    """
    cut_result = []
    for i in fft_data :
        cut_result.append(np.max(i))

    
    peaks,_ = scipy.signal.find_peaks(cut_result)
    #print(peaks)


    #print(point_cut)

    tmp_data = []
    for i in peaks :
        if np.min(np.abs(point_cut - i)) <= 3 :
            tmp_data.append(i)
    print(tmp_data)
    #draw_single_lines(np.array(cut_result),r"F:\work\video_analyze\output","cut_reslut")

    with open(r"F:\work\video_analyze\output\lnc_time.txt","r") as txt_file :
        line = txt_file.readline()
        lnc_time = list(map(float,line.replace("\r","").split("\t")))
    print(lnc_time)
    
    x_bar = np.arange( 0 , len( cut_result ) , 1 ) / fps
    #peaks = scipy.signal.find_peaks_cwt(input_data,5)
    
    fig = plt.figure(figsize=(70,8))
    plt.subplot(111)
    plt.plot( x_bar , cut_result , c = "r" )
    #plt.scatter(tmp_data/fps, np.array(cut_result)[tmp_data])
    plt.vlines(x_bar[vocal_cut], 0, max(cut_result), alpha=0.5, color='r',linestyle='--', label='Beats')
    plt.vlines(lnc_time, 0, max(cut_result), alpha=0.5, color='b',linestyle='--', label='lnc')
    plt.savefig(os.path.join(r"F:\work\video_analyze\output","{}.png".format("cut_reslut")),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
    """


    """
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    #輸出影像設定
    out = cv2.VideoWriter( os.path.join('test.avi'), fourcc, fps, (240,240))   #輸出無聲長條影像
    
    
    for i in tqdm(range(int(len(y)/sr*fps))):
        img = np.zeros(( 240,240, 3), np.uint8) #製造空畫面
        if i in vocal_cut :
            img.fill(200)   #填滿單色
        else :
            img.fill(0)
        #img.fill(200)
        #cv2.putText(img, str(i) , (10, 230), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255), 1, cv2.LINE_AA)
        out.write(img)
    out.release()   #清理記憶體
    cv2.destroyAllWindows()


    
    #audioclip = AudioFileClip(r"F:\work\video_analyze\data\audio\Beelzebub-jou no Okinimesu Mama\separated\htdemucs\01.ピンクレモネード\vocals.wav") #獲取音頻
    audioclip = AudioFileClip(audio_path)
    clip = VideoFileClip( os.path.join(r"test.avi"))    #獲取影片
    new_video = clip.set_audio(audioclip)   #影片合併音頻
    new_video.write_videofile( os.path.join(r"test.mp4"))    #輸出影片
    
    """

    """
    fig, ax = plt.subplots( figsize = (20,8) , nrows=2, sharex=True)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max),y_axis='mel', x_axis='time', hop_length=hop_length,ax=ax[0])
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_env),
            label='Onset strength')
    ax[1].vlines(times[tmp_beats], 0, 1, alpha=0.5, color='r',linestyle='--', label='Beats')
    ax[1].legend()
    #plt.savefig("%s.png"%(i.replace(".wav","")))
    plt.savefig("test.png")
    plt.close('all')
    """

def check_list(lst):
    if True in lst:
        return True, [i for i, x in enumerate(lst) if x]
    else:
        return False,[]

def min_index(lst):
    min_val = min(lst)
    return [i for i, x in enumerate(lst) if x == min_val]

def max_index(lst):
    max_val = max(lst)
    return [i for i, x in enumerate(lst) if x == max_val]

def fill_fields(fields, objects_dict):

    origin_fields = fields.copy()
    objects = list(objects_dict.keys())


    rec_dict = {}
    obj_idx = 0
    while obj_idx < len(objects) and max(fields) != 0 :
    #for i,x in enumerate(objects) :
        x = objects[obj_idx]
        gap = np.array(fields)-int(x)
        T_F , ind_list = check_list(gap>=0)
        print(fields)
        #print(rec_dict)
        if T_F and x in objects_dict.keys() :
            rec_idx = ind_list[min_index([ gap[k] for k in ind_list])[0]]
            if str(rec_idx) not in rec_dict.keys() : rec_dict[str(rec_idx)]  = []
            rec_dict[str(rec_idx)].append([x,x])
            fields[rec_idx] -= x
            objects_dict[x] -= 1
            if objects_dict[x] == 0 : del objects_dict[x]
        else :
            obj_idx += 1


    print(origin_fields)
    print(fields)
    print(rec_dict)
    #######################################################################################
    for idx,ele in enumerate(fields) :
        if (ele != 0) and (str(idx) in rec_dict.keys()):
            len_list = rec_dict[str(idx)]
            sum_len = sum([i[0] for i in len_list])
            magnification_len = (sum_len + ele)/sum_len
            if magnification_len <= 1.2 :
                for rec_val in range(len(rec_dict[str(idx)])) :
                    error_len = int(ele*rec_dict[str(idx)][rec_val][0]/sum_len+0.5)
                    rec_dict[str(idx)][rec_val][1] = rec_dict[str(idx)][rec_val][0] + error_len
                    fields[idx] -= error_len
                    if fields[idx] < 0 :
                        rec_dict[str(idx)][rec_val][1] += fields[idx]
                        fields[idx] = 0
                """
                for rec_val in rec_dict[str(idx)] :
                    rec_val[1] = (sum_len + ele)/sum_len
                rec_dict[str(idx)][max_idx][1] = max_len + ele
                fields[idx] = 0
                """
            max_idx = max_index([i[0] for i in len_list])[0]
            max_len = len_list[max_idx][0]
            if (max_len + ele) / max_len <= 1.2 :
                rec_dict[str(idx)][max_idx][1] = max_len + ele
                fields[idx] = 0
    print(fields)
    print(rec_dict)
    #######################################################################################
    
    if max(fields) != 0 and  objects_dict != {} :
        objects = list(objects_dict.keys())
        for idx,ele in enumerate(fields) :
            if ele != 0 :
                gap = np.array(objects)-int(ele)
                min_obj = objects[min_index(gap)[0]]
                if str(idx) not in rec_dict.keys() : rec_dict[str(idx)]  = []
                rec_dict[str(rec_idx)].append([min_obj,ele])
                fields[idx] -= ele
                objects_dict[min_obj] -= 1
                if objects_dict[min_obj] == 0 : del objects_dict[min_obj]
    
    if max(fields) != 0 : raise Exception( "Cut too little {}".format(fields) )
    return rec_dict




def get_point(audio_file):
    # 讀取音訊檔案
    y_audio, sr_audio = librosa.load(audio_file)

    # 設定窗口大小和跨度
    frame_length = 64
    hop_length = 512

    # 計算短時能量
    energy = librosa.feature.rms(y=y_audio, frame_length=frame_length, hop_length=hop_length)[0]
    ZCR = librosa.feature.zero_crossing_rate(y=y_audio, frame_length=frame_length, hop_length=hop_length)[0]
    print(len(y_audio))
    print(sr_audio)
    print(len(energy))
    print(energy)
    print(len(ZCR))

    x_bar = np.arange( 0 , len( energy ) , 1 )
    fig = plt.figure(figsize=(70,8))
    plt.subplot(111)
    plt.plot( x_bar , ZCR , c = "b" )
    plt.plot( x_bar , energy , c = "r" )
    #plt.scatter(tmp_data/fps, np.array(cut_result)[tmp_data])
    plt.savefig(os.path.join(r"F:\work\video_analyze\output","{}.png".format("cut_reslut_2")),bbox_inches='tight',pad_inches = 0)
    plt.close('all')



    # 設定閾值
    threshold = 0.02

    # 尋找開始發聲的位置
    onset_frames = librosa.onset.onset_detect(y=y_audio, sr=sr_audio, hop_length=hop_length, backtrack=True, energy=ZCR, 
                                            units='frames', pre_max=1, post_max=1, pre_avg=1, post_avg=1, 
                                            delta=0.15, wait=10)

    # 將帧位置轉換為秒
    onset_times = librosa.frames_to_time(onset_frames, sr=sr_audio, hop_length=hop_length)

    #clustering = DBSCAN(eps=10, min_samples=1).fit(onset_frames.reshape(-1, 1))
    kmeans_fit = cluster.KMeans(n_clusters = 59).fit(onset_frames.reshape(-1, 1))
    #print(clustering.labels_)
    #print(kmeans_fit.labels_)

    used_label = []
    return_onset_times = []
    for i in range(len(onset_frames)) :
        if kmeans_fit.labels_[i] not in used_label and kmeans_fit.labels_[i] != -1 :
            used_label.append(kmeans_fit.labels_[i])
            return_onset_times.append(onset_times[i])

    # 顯示開始發聲的位置
    #print('The onset frames are:', onset_frames)
    #print('The onset times are:', onset_times)

    return return_onset_times

def calVolume(waveData, frameSize, overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = int(math.ceil(wlen*1.0/step))
    volume = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]
        curFrame = curFrame - np.median(curFrame) # zero-justified
        volume[i] = np.sum(np.abs(curFrame))
    return volume


def calVolumeDB(waveData, frameSize, overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = int(math.ceil(wlen*1.0/step))
    volume = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]
        curFrame = curFrame - np.mean(curFrame) # zero-justified
        volume[i] = 10*np.log10(np.sum(curFrame*curFrame))
    return volume


def get_cut_video_dict():

    cut_video_path = r"F:\work\video_analyze\output\cut_video\Detective Conan The Culprit Hanzawa"
    with open(r"F:\work\video_analyze\Hanzawa_video_dict.yaml","r") as yaml_file :
        video_dict = yaml.load(yaml_file,Loader=yaml.Loader)

    pick_video = []
    for key,value in video_dict.items() :
        if not os.path.exists(os.path.join(cut_video_path,"{}.mp4".format(key))) : 
            print(os.path.join(cut_video_path,key))
            continue
        tmp_dict = video_dict[key]
        #pick_score = max((tmp_dict["charater"]/tmp_dict["frame_num"] >= 1 ),(tmp_dict["creature"] > 0))*( tmp_dict["text"] == 0 )
        gray_mean = tmp_dict["gray_mean"]
        gray_std = tmp_dict["gray_std"]
        #pick_score = (tmp_dict["Beelzebub"]/tmp_dict["frame_num"] >= 0 )*( tmp_dict["text"] == 0 )*( tmp_dict["title"] == 0 )*( min(gray_mean) >= 150 )*( min( gray_std ) >= 15 )
        pick_score = (tmp_dict["black"] >= 0 )*( tmp_dict["text"] == 0 )*( tmp_dict["title"] == 0 )*( min(gray_mean) >= 100 )*( min( gray_std ) >= 15 )
        if pick_score : pick_video.append([key,tmp_dict["frame_num"]])

    return_dict = {}
    pattern = r"(.+?)_(\d+)"
    for i in pick_video :
        match = re.search(pattern, i[0])
        frame_count = i[1]
        if match:
            name = match.group(1)
            num = match.group(2)
            if frame_count not in return_dict.keys() : return_dict[frame_count] = []
            return_dict[frame_count].append([name,num])
        else : print("No match")
    
    return return_dict



"""
return_dict = {}
pattern = r"(.+?)_(\d+)\.mp4$"
for i in os.listdir(video_path) :
    vidCap = cv2.VideoCapture(os.path.join(video_path,i))
    match = re.search(pattern, i)
    # 提取匹配的結果
    if match:
        name = match.group(1)
        num = match.group(2)
        frame_count = str(int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count not in return_dict.keys() : return_dict[frame_count] = []
        return_dict[frame_count].append([name,num])
    else:
        print("No match")

    # 釋放資源
    vidCap.release()

return return_dict
"""


if __name__ == "__main__" :
    main()
    #audio_path = r"F:\work\video_analyze\separated\htdemucs\test\vocals.wav"
    #get_point(audio_path)


    #data_dict = get_cut_video_dict(r"F:\work\video_analyze\output\cut_video")
    #print(data_dict)



    #print(fill_fields([10,9,8,7],[8,5,2,1]))

    """
    fw = wave.open(audio_path,'r')
    params = fw.getparams()
    print(params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = fw.readframes(nframes)
    waveData = np.fromstring(strData, dtype=np.int16)
    waveData = waveData*1.0/max(abs(waveData))  # normalization
    fw.close()

    # calculate volume
    frameSize = 256
    overLap = 128
    volume11 = calVolume(waveData,frameSize,overLap)
    volume12 = calVolumeDB(waveData,frameSize,overLap)

    # plot the wave
    time = np.arange(0, nframes*2)*(1.0/framerate)
    time2 = np.arange(0, len(volume11))*(frameSize-overLap)*1.0/framerate
    pl.subplot(311)
    pl.plot(time, waveData)
    pl.ylabel("Amplitude")
    pl.subplot(312)
    pl.plot(time2, volume11)
    pl.ylabel("absSum")
    pl.subplot(313)
    pl.plot(time2, volume12, c="g")
    pl.ylabel("Decibel(dB)")
    pl.xlabel("time (seconds)")
    pl.show()
    """