import librosa
import numpy as np
from Helper_private import get_fft,shorten_number,dataframe_change,get_hist,get_list_barycenter
import cv2
import os
import matplotlib.pyplot as plt
import torch
import time
import yaml
import threading
from tqdm import tqdm
import threading
import copy

def min_index(lst):
    min_val = min(lst)
    return [i for i, x in enumerate(lst) if x == min_val]



def get_paragraph(audio_path,vocal_path,fps,lnc_time_path):
    paragraph_dict = {
        "paragraph":[],
        "interlude" : [],
        "vocal" : {}
    }
    ##########################################################################################################################
    y,sr = librosa.load(audio_path)
    fft_data = get_fft(audio_path,fps)
    totla_frame = int(len(y)/sr*fps)
    ##########################################################################################################################
    vocal_frams = get_point(vocal_path,fps)
    print(vocal_frams)
    ##########################################################################################################################
    with open(lnc_time_path,"r") as txt_file :
        line = txt_file.readline()
        lnc_time = np.array(list(map(float,line.replace("\r","").split("\t"))))
    vocal_cut = np.array((lnc_time-0.3)*fps)
    ##########################################################################################################################
    tmp_vocal_frams = np.array([ i[0] for i in vocal_frams])
    for idx,ele in enumerate(vocal_cut) :
        dist = np.abs(tmp_vocal_frams-ele)
        if min(dist) < 3*fps :
            vocal_cut[idx] = tmp_vocal_frams[min_index(dist)[0]]
    vocal_cut = np.array(vocal_cut,dtype="uint64")
    print( vocal_cut )
    ##########################################################################################################################
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
    ##########################################################################################################################
    if interlude[0][0] < vocal_frams[0][0] :inter_vocal = ["interlude","vocal"]
    else : inter_vocal = ["vocal","interlude"]
    iv_count = [0,0]
    for i in range( len( vocal_frams ) + len(interlude ) ) :
        paragraph_dict["paragraph"].append( (inter_vocal[i%2],iv_count[i%2]) )
        iv_count[i%2] += 1
    ##########################################################################################################################
    for i in interlude :
        paragraph_dict["interlude"].append( i[1]-i[0] )
    vocal_frame_idx = 0
    for vocal_frame_idx in range(len(vocal_frams)) :
        tmp = [vocal_frams[vocal_frame_idx][0]] + [ i for i in vocal_cut if (i > vocal_frams[vocal_frame_idx][0] and i < vocal_frams[vocal_frame_idx][1]) ] + [vocal_frams[vocal_frame_idx][1]]
        paragraph_dict["vocal"][vocal_frame_idx] = []
        for i in range(len(tmp)-1) :
            paragraph_dict["vocal"][vocal_frame_idx] .append( tmp[i+1] - tmp[i] )
    ##########################################################################################################################
    return paragraph_dict



def get_cut_video(var_hist_path,video_name,fps) :
    
    ##########################################################################################################################
    #start_point = int(fps*(1*60+40))
    #end_point = int(fps*(9*60+10))

    min_sec_set = 3
    min_interval_set = fps*min_sec_set
    #interval_set = fps*sec_set

    var_hist = []
    with open(os.path.join(var_hist_path,"%s_hist_var.txt"%(video_name)),"r") as txtfile :
        lines = txtfile.readlines()
        for i in lines :
            var_hist.append(eval(i))
    #start_point = int(fps*(10*60))
    #end_point = int(fps*(20*60))
    start_point = int(fps*(1*60+50))
    end_point = int(fps*(9*60))
                    
    cut_point_dict = {}
    cut_point_idx = 0
    rec_state = False
    tmp_hist = []
    for i in tqdm(range(start_point,end_point)) :
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
    vocal_frames = []
    for idx,ele in enumerate(energy_gap_con) :
        if ele == 1 and not if_rec :
            if_rec = True
            start_frame = energy_frame[idx]
        elif ( ele == 0 or idx == len(energy_gap_con)-1 ) and if_rec :
            if_rec = False
            end_frame =  energy_frame[idx]
            vocal_frames.append( [start_frame,end_frame] )


    
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
    

    # 尋找開始發聲的位置
    onset_frames = librosa.onset.onset_detect(y=y_audio, sr=sr_audio, hop_length=hop_length, backtrack=True, energy=ZCR, 
                                            units='frames', pre_max=1, post_max=1, pre_avg=1, post_avg=1, 
                                            delta=0.15, wait=10)

    # 將帧位置轉換為秒
    onset_times = librosa.frames_to_time(onset_frames, sr=sr_audio, hop_length=hop_length)


    return vocal_frames


def video_predict(video_path,predict_model,cut_point_dict) :
    #class_num = ["creature","text","Beelzebub","title"]
    class_num = ["black","text","title"]
    vidCap = cv2.VideoCapture(video_path)
    video_length = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))

    return_dict_ori = {
        "gray_mean" : [],
        "gray_std" : [],
        "BGR" : {
            0:[],
            1:[],
            2:[]
        }
    }
    for i in class_num :
        return_dict_ori[i] = 0
    return_dict = copy.deepcopy(return_dict_ori)
    current_frame = 0
    cpd_key = 0
    while True :
        start_time = time.time()
        if cpd_key not in cut_point_dict.keys() : break
        if current_frame >= cut_point_dict[cpd_key]["interval"][1] :
            for key,val in return_dict.items() :
                cut_point_dict[cpd_key][key] = val
            return_dict = copy.deepcopy(return_dict_ori)
            cpd_key += 1
            continue
        ret = vidCap.grab()
        if not ret: break
        if current_frame >= cut_point_dict[cpd_key]["interval"][0] :
            ret,image = vidCap.retrieve()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ##########################################################################################
            return_dict["gray_mean"].append( shorten_number(np.mean(gray_image)))
            return_dict["gray_std"].append(shorten_number(np.std(gray_image)))
            ##########################################################################################
            for channal_num in range(3) :
                return_dict["BGR"][channal_num].append(get_list_barycenter(get_hist(image,channal_num)))
            ##########################################################################################
            predict_result = predict_model(image)
            data_frame = dataframe_change(predict_result.pandas().xyxy)[0]
            """
            for i in data_frame :
                if i[-1] == 0 :
                    if i[-2] >= 0.6 : return_dict["creature"] += 1
                elif i[-1] == 2 :
                    if i[-2] >= 0.95 : return_dict["Beelzebub"] += 1
                else : return_dict[class_num[int(i[-1])]] += 1
            """
            for i in data_frame :
                if i[-1] == 0 :
                    if i[-2] >= 0.8 : return_dict["black"] += 1
                else : return_dict[class_num[int(i[-1])]] += 1
            ##########################################################################################
            end_time = time.time()
            print("{}/{} --- {:.2f} it/s.".format(current_frame,video_length,1/(end_time-start_time)),end = "\r")
        current_frame += 1
        
    vidCap.release()
    cv2.destroyAllWindows()

    return cut_point_dict


def task(base_path,file_name,var_hist_path,predict_model,save_path,fps) :
    cut_video_data = {}
    video_name = file_name.replace(".mp4","")
    video_path = os.path.join( base_path , file_name )
    cut_point_dict = get_cut_video(var_hist_path,video_name,fps)
    cut_point_dict = video_predict( video_path , predict_model , cut_point_dict  )
    cut_video_data[video_name] = cut_point_dict
    with open(os.path.join(save_path,'%s.yaml'%(video_name)), 'w') as f:
        yaml.dump(cut_video_data, f)

class MyThread(threading.Thread):
    def __init__(self, base_path,file_name,var_hist_path,predict_model,save_path,fps,semaphore):
        threading.Thread.__init__(self)
        self.base_path = base_path
        self.file_name = file_name
        self.var_hist_path = var_hist_path
        self.predict_model =predict_model
        self.save_path = save_path
        self.fps = fps
        self.semaphore = semaphore
    ####################################################################################################################################
    def run(self):
        with self.semaphore :
            task(self.base_path,self.file_name,self.var_hist_path,self.predict_model,self.save_path,self.fps)
####################################################################################################################################
if __name__ == "__main__" :
    video_name = ""
    title = "Detective Conan The Culprit Hanzawa"
    #title = "Beelzebub-jou no Okinimesu mama"
    var_hist_path = "F:\\work\\video_analyze\\output\\var_hist\\%s"%(title)
    base_path = "F:\\work\\video_analyze\\data\\video\\%s"%(title)
    save_path = "F:\\work\\video_analyze\\output\\cut_video_data\\%s"%(title)
    audio_path = r"F:\work\video_analyze\data\audio\Detective Conan The Culprit Hanzawa\01.捕まえて、今夜。.flac"
    vocal_path = r"F:\work\video_analyze\data\audio\Detective Conan The Culprit Hanzawa\separated\htdemucs\01.捕まえて、今夜。\vocals.wav"
    lnc_time_path = r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa\Hanzawa_lnc_time.txt"
    fps = 23.976
    ####################################################################################################################################
    paragraph_dict = get_paragraph(audio_path,vocal_path,fps,lnc_time_path)
    print(paragraph_dict)
    
    with open(os.path.join(save_path,'paragraph_dict.yaml'), 'w') as f:
        yaml.dump(paragraph_dict, f)
    ####################################################################################################################################
    """
    device = torch.device("cuda:0")
    predict_model = torch.hub.load('ultralytics/yolov5', 'custom', path = r"F:\work\yolov5\runs\train\exp9\weights\best.pt")
    predict_model.iou = 0.2
    predict_model.conf = 0.3
    predict_model.to(device)
    ####################################################################################################################################
    max_deals = 10
    semaphore = threading.BoundedSemaphore(max_deals)
    threads = []
    for file_name in os.listdir(base_path) :
        threads.append(MyThread(base_path,file_name,var_hist_path,predict_model,save_path,fps,semaphore))
    ####################################################################################################################################
    for _idx_ in range(len(threads)) :
        threads[_idx_].start()
    ####################################################################################################################################
    for _idx_ in range(len(threads)):
        threads[_idx_].join()
    """
    