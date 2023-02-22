import librosa
import numpy as np
import librosa.display
from Helper_private import *
from sklearn.cluster import DBSCAN
from sklearn import cluster
import threading


class MyThread(threading.Thread):
    def __init__(self, y_vocal, sr_vocal,frame_length,hop_length,delta,knum,timestep,semaphore):
        threading.Thread.__init__(self)
        self.y_vocal = y_vocal
        self.sr_vocal = sr_vocal
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.delta = delta
        self.knum = knum
        self.timestep = timestep
        self.semaphore = semaphore
    ####################################################################################################################################
    def run(self):
        with self.semaphore :
            get_point(self.y_vocal,self.sr_vocal,self.frame_length,self.hop_length,self.delta,self.knum,self.timestep)


def main():
    data = []
    with open(r"F:\work\video_analyze\data\timestep.txt","r",encoding="utf8") as txtfile :
        lines = txtfile.readlines()
        for i in lines :
            data.append(i.replace("\n","").split("\t"))
    
    timestep = np.array([ int(i[0]) for i in data ])
    vocal_path = r"F:\work\video_analyze\data\audio\Beelzebub-jou no Okinimesu Mama\separated\htdemucs\01.ピンクレモネード\vocals.wav"
    y_vocal, sr_vocal = librosa.load(vocal_path)
    """
    for fr_len in range(10,20) :
        for hp_le in [256,512,1024] :
            for dta in range(1,6) :
                vocal_cut = get_point(y_vocal, sr_vocal,frame_length=fr_len,hop_length=hp_le,delta=dta/10,knum=len(timestep))
                if len(vocal_cut) == len(timestep) :
                    score = single_distance(np.abs(timestep-vocal_cut))

    """

    max_deals = 16
    semaphore = threading.BoundedSemaphore(max_deals)
    threads = []
    for fr_len in range(10,2048) :
        for hp_le in [256,512,1024] :
            for dta in range(1,100) :
                threads.append(MyThread(y_vocal,sr_vocal,fr_len,hp_le,dta/100,len(timestep),timestep,semaphore))
    ####################################################################################################################################
    for _idx_ in range(len(threads)) :
        threads[_idx_].start()
    ####################################################################################################################################
    for _idx_ in range(len(threads)):
        threads[_idx_].join()




def distance(input_data):
    return np.power(np.sum([ i*i for i in input_data ]),1/2)


def get_point(y_vocal, sr_vocal,frame_length:int,hop_length:int,delta:float,knum:int,timestep):
    
    print( frame_length,hop_length,delta,knum,end="\r" )
    
    # 計算短時能量
    energy = librosa.feature.rms(y=y_vocal, frame_length=frame_length, hop_length=hop_length)[0]


    # 尋找開始發聲的位置
    onset_frames = librosa.onset.onset_detect(y=y_vocal, sr=sr_vocal, hop_length=hop_length, backtrack=False, energy=energy, 
                                            units='frames', pre_max=1, post_max=1, pre_avg=1, post_avg=1, 
                                            delta=delta, wait=0)

    # 將帧位置轉換為秒
    onset_times = librosa.frames_to_time(onset_frames, sr=sr_vocal, hop_length=hop_length)

    if len(onset_frames) >= knum :

        #clustering = DBSCAN(eps=10, min_samples=3).fit(onset_frames.reshape(-1, 1))
        kmeans_fit = cluster.KMeans(n_clusters = knum).fit(onset_frames.reshape(-1, 1))
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
        fps = 23.98
        vocal_cut = np.array(np.array(return_onset_times)*fps,dtype="uint64")
        
        score = single_distance(np.abs(timestep-vocal_cut))

        with open(r"output\score.txt","a") as txtfile :
            return_list = [ frame_length , hop_length , delta , score ]
            print( "\t".join( list(map(str,return_list) )) , file = txtfile )




if __name__ == "__main__" :
    main()