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
def main() :
    fps = 30
    #audio_path = os.path.join(r"F:\work\video_analyze\separated\htdemucs\test",i)
    audio_path = r"F:\work\video_analyze\data\test.mp3"
    y,sr = librosa.load(audio_path)
    


    vocal_path = r"F:\work\video_analyze\separated\htdemucs\test\vocals.wav"
    vocal_point = get_point(vocal_path)
    vocal_cut = np.array(np.array(vocal_point)*fps,dtype="uint64")

    #print(y.shape)
    #print(sr)

    tempo,beats = librosa.beat.beat_track(y=y,sr=sr)


    #print(tempo,beats)

    #print(librosa.frames_to_time(beats, sr=sr))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr,aggregate=np.median)


    

    tmp_beats = []
    for i in range(len(beats)):
        if i%4 == 0 :
            tmp_beats.append(beats[i])
    tmp_time = librosa.frames_to_time(tmp_beats, sr=sr)
    point_cut = np.array(np.array(tmp_time)*fps,dtype="uint64")


    fft_data = get_fft(audio_path,fps)
    
    cut_result = []
    for i in fft_data :
        cut_result.append(np.max(i[:100]))

    
    peaks,_ = scipy.signal.find_peaks(cut_result)
    #print(peaks)


    #print(point_cut)

    tmp_data = []
    for i in peaks :
        if np.min(np.abs(point_cut - i)) <= 3 :
            tmp_data.append(i)
    print(tmp_data)
    #draw_single_lines(np.array(cut_result),r"F:\work\video_analyze\output","cut_reslut")
    
    x_bar = np.arange( 0 , len( cut_result ) , 1 )
    #peaks = scipy.signal.find_peaks_cwt(input_data,5)
    
    fig = plt.figure(figsize=(70,8))
    plt.subplot(111)
    plt.plot( x_bar , cut_result , c = "r" )
    plt.scatter(tmp_data, np.array(cut_result)[tmp_data])
    plt.vlines(x_bar[vocal_cut], 0, max(cut_result), alpha=0.5, color='r',linestyle='--', label='Beats')
    plt.savefig(os.path.join(r"F:\work\video_analyze\output","{}.png".format("cut_reslut")),bbox_inches='tight',pad_inches = 0)
    plt.close('all')

    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    #輸出影像設定
    out = cv2.VideoWriter( os.path.join('test.avi'), fourcc, fps, (240,240))   #輸出無聲長條影像
    
    
    for i in tqdm(range(int(len(y)/sr*fps))):
        img = np.zeros(( 240,240, 3), np.uint8) #製造空畫面
        if i in vocal_cut :
            img.fill(200)   #填滿單色
        else :
            img.fill(0)
        out.write(img)
    out.release()   #清理記憶體
    cv2.destroyAllWindows()


    
    audioclip = AudioFileClip(r"F:\work\video_analyze\data\test.mp3") #獲取音頻
    clip = VideoFileClip( os.path.join(r"test.avi"))    #獲取影片
    new_video = clip.set_audio(audioclip)   #影片合併音頻
    new_video.write_videofile( os.path.join(r"test.mp4"))    #輸出影片

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

def get_point(audio_file):
    # 讀取音訊檔案
    y, sr = librosa.load(audio_file)

    # 設定窗口大小和跨度
    frame_length = 1024
    hop_length = 512

    # 計算短時能量
    energy = librosa.feature.rms(y, frame_length=frame_length, hop_length=hop_length)[0]


    # 設定閾值
    threshold = 0.02

    # 尋找開始發聲的位置
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=False, energy=energy, 
                                            units='frames', pre_max=1, post_max=1, pre_avg=1, post_avg=1, 
                                            delta=0.12, wait=0)

    # 將帧位置轉換為秒
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


    clustering = DBSCAN(eps=30, min_samples=1).fit(onset_frames.reshape(-1, 1))
    kmeans_fit = cluster.KMeans(n_clusters = 41).fit(onset_frames.reshape(-1, 1))
    #print(clustering.labels_)
    #print(kmeans_fit.labels_)

    used_label = []
    return_onset_times = []
    for i in range(len(onset_frames)) :
        if clustering.labels_[i] not in used_label and clustering.labels_[i] != -1 :
            used_label.append(clustering.labels_[i])
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


if __name__ == "__main__" :
    main()
    #audio_path = r"F:\work\video_analyze\separated\htdemucs\test\vocals.wav"
    #get_point(audio_path)



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