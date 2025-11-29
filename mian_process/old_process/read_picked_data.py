import os
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import AudioFileClip,VideoFileClip
def get_interval_data(video_name,cut_idx) :
    data_path = "F:\\work\\video_analyze\\output\\cut_video_data\\Detective Conan The Culprit Hanzawa\\%s.yaml"%(video_name)
    #data_path = "F:\\work\\video_analyze\\output\\cut_video_data\\Beelzebub-jou no Okinimesu mama\\%s.yaml"%(video_name)
    with open( data_path,"r" ) as yamlfile :
        data_dict = yaml.load(yamlfile,Loader=yaml.Loader)
    return data_dict[video_name][cut_idx]["interval"]

def cross_function(lenght) :
    return np.linspace( 0,1,lenght+1 )
    #return x/lenght

def wirte_in_list( cut_data_name,interval,target_len ) :
    video_folder = r"F:\work\video_analyze\data\video\Detective Conan The Culprit Hanzawa"
    #video_folder = r"F:\work\video_analyze\data\video\Beelzebub-jou no Okinimesu mama"
    video_path = os.path.join(video_folder,"{}.mp4".format(cut_data_name))
    vidCap = cv2.VideoCapture(video_path)
    start_frame = interval[0]
    end_frame = interval[1]
    length = end_frame - start_frame
    speed_spot = np.linspace(0,length,length+1) * max(length,target_len) / length
    frame_num = 0
    speed_spot_idx = 0
    write_idx = 0
    return_list = []
    while True :
        if frame_num < start_frame :
            ret = vidCap.grab()
            frame_num += 1
            continue
        if frame_num < min(end_frame,start_frame+target_len) :
            if write_idx >= speed_spot[speed_spot_idx] :
                ret = vidCap.grab()
                if not ret : break
                ret,image = vidCap.retrieve()
                frame_num += 1
                speed_spot_idx += 1
            return_list.append(image)
            write_idx += 1
        else : break
    vidCap.release()
    return return_list


def wirte_in( out,cut_data_name,interval,target_len ) :
    #video_folder = r"F:\work\video_analyze\data\video\Detective Conan The Culprit Hanzawa"
    video_folder = r"F:\work\video_analyze\data\video\Beelzebub-jou no Okinimesu mama"
    video_path = os.path.join(video_folder,"{}.mp4".format(cut_data_name))
    vidCap = cv2.VideoCapture(video_path)
    start_frame = interval[0]
    end_frame = interval[1]
    length = end_frame - start_frame
    speed_spot = np.linspace(0,length,length+1) * max(length,target_len) / length
    frame_num = 0
    speed_spot_idx = 0
    write_idx = 0
    while True :
        if frame_num < start_frame :
            ret = vidCap.grab()
            frame_num += 1
            continue
        if frame_num < min(end_frame,start_frame+target_len) :
            if write_idx >= speed_spot[speed_spot_idx] :
                ret = vidCap.grab()
                if not ret : break
                ret,image = vidCap.retrieve()
                frame_num += 1
                speed_spot_idx += 1
            out.write(image)
            write_idx += 1
        else : break
    vidCap.release()
    return out

def main2() :
    base_path = r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa"
    #base_path = r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama"
    with open( os.path.join(base_path,"picked_data.yaml"),"r" ) as yamlfile :
        paragraph_dict = yaml.load(yamlfile,Loader=yaml.Loader)


    total_process = []
    for target_par in paragraph_dict["paragraph"] :
        (iv_key,idx_key) = target_par
        if iv_key == 'interlude' :
            process_data = paragraph_dict["interlude_data"][idx_key]
            target_len = paragraph_dict[iv_key][idx_key]+10
            cut_len = sum([ i[-1]-10 for i in process_data ])+10
            if cut_len < target_len :
                print(process_data)
                print("1:",cut_len, target_len)
                chage_rate = target_len / cut_len
                for i in process_data :
                    total_process.append(i + [ int((i[-1]-10) * chage_rate+0.5)+10 ])
            elif cut_len >= target_len :
                print("2:",cut_len, target_len)
                for i in process_data :
                    print(i[-1], target_len)
                    total_process.append(i + [ min( i[-1] , target_len ) ])
                    target_len = target_len - min( i[-1] , target_len )
                    if target_len == 0 : break
                    target_len += 10
        elif iv_key == 'vocal' :
            process_data = paragraph_dict["vocal_data"][idx_key]
            target_len_list = paragraph_dict[iv_key][idx_key]
            for idx,target_len in enumerate(target_len_list) :
                cut_len = process_data[idx][-1]
                total_process.append( process_data[idx] + [ target_len+10 ])
    target_len_list = [i[-1] for i in total_process]

    
    space_list = [ []  for _ in range(int(sum( target_len_list )-(len(target_len_list)-1)*10))]

    start_point = 0
    for idx,ele in enumerate( target_len_list ) :
        print("get pic process : {} / {} ".format(idx+1,len(target_len_list)),end="\r")
        #print( int(start_point),int(start_point+ele))
        target_process = total_process[idx]
        wi_list = wirte_in_list( target_process[0],get_interval_data(target_process[0],target_process[1]),ele )
        for i in range( int(start_point),int(start_point+ele) ) :
            space_list[i].append( wi_list[int(i-start_point)] )
            #space_list[i].append( target_process[0] )
        start_point += ele - 10
    print("\n")
    ####################################################################################################################################################
    fps = 23.976
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    #輸出影像設定
    #out = cv2.VideoWriter( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama\Beelzebub-jou no Okinimesu mam.avi"), fourcc, fps, (1280,720))   #輸出無聲長條影像
    out = cv2.VideoWriter( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa\Detective Conan The Culprit Hanzawa.avi"), fourcc, fps, (1920,1080))   #輸出無聲長條影像
    r_per = 0
    cross_y =  cross_function(10)
    for idx,ele in enumerate(space_list[5:-5]) :
        print("writing process : {} / {} ".format(idx+1,len(space_list[5:-5])),end="\r")
        if len(ele) == 1 :
            r_per = 0
            out.write(ele[0])
        elif len(ele) == 2 :
            #print( idx , r_per , 1 - cross_y[r_per], cross_y[r_per] )
            out.write(cv2.addWeighted( ele[0] , 1 - cross_y[r_per] , ele[1], cross_y[r_per] ,0 ))
            r_per += 1

    out.release()   #清理記憶體
    cv2.destroyAllWindows()
    
    audio_path = r"F:\work\video_analyze\data\audio\Detective Conan The Culprit Hanzawa\01.捕まえて、今夜。.flac"
    #audio_path = r"F:\work\video_analyze\data\audio\Beelzebub-jou no Okinimesu Mama\01.ピンクレモネード.wav"
    audioclip = AudioFileClip(audio_path)
    #clip = VideoFileClip( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama\Beelzebub-jou no Okinimesu mam.avi"))    #獲取影片
    clip = VideoFileClip( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa\Detective Conan The Culprit Hanzawa.avi"))    #獲取影片
    new_video = clip.set_audio(audioclip)   #影片合併音頻
    #new_video.write_videofile( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama\Beelzebub-jou no Okinimesu mam.mp4")) 
    new_video.write_videofile( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa\Detective Conan The Culprit Hanzawa.mp4")) 

def main() :
    #base_path = r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa"
    base_path = r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama"
    with open( os.path.join(base_path,"picked_data.yaml"),"r" ) as yamlfile :
        paragraph_dict = yaml.load(yamlfile,Loader=yaml.Loader)

    total_process = []
    for target_par in paragraph_dict["paragraph"] :
        (iv_key,idx_key) = target_par
        if iv_key == 'interlude' :
            process_data = paragraph_dict["interlude_data"][idx_key]
            target_len = paragraph_dict[iv_key][idx_key]
            cut_len = sum([ i[-1] for i in process_data ])
            if cut_len < target_len :
                chage_rate = target_len / cut_len
                for i in process_data :
                    total_process.append(i + [ int(i[-1] * chage_rate +0.5) ])
            elif cut_len >= target_len :
                for i in process_data :
                    total_process.append(i + [ min( i[-1] , target_len ) ])
                    target_len = target_len - min( i[-1] , target_len )
        elif iv_key == 'vocal' :
            process_data = paragraph_dict["vocal_data"][idx_key]
            target_len_list = paragraph_dict[iv_key][idx_key]
            for idx,target_len in enumerate(target_len_list) :
                cut_len = process_data[idx][-1]
                total_process.append( process_data[idx] + [ target_len ])


    fps = 23.976
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    #輸出影像設定
    out = cv2.VideoWriter( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama\Beelzebub-jou no Okinimesu mam.avi"), fourcc, fps, (1280,720))   #輸出無聲長條影像

    for i in tqdm(total_process) :
        out = wirte_in( out,i[0],get_interval_data(i[0],i[1]),i[3] )
    out.release()   #清理記憶體
    cv2.destroyAllWindows()

    
    #audio_path = r"F:\work\video_analyze\data\audio\Detective Conan The Culprit Hanzawa\01.捕まえて、今夜。.flac"
    audio_path = r"F:\work\video_analyze\data\audio\Beelzebub-jou no Okinimesu Mama\01.ピンクレモネード.wav"
    audioclip = AudioFileClip(audio_path)
    clip = VideoFileClip( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama\Beelzebub-jou no Okinimesu mam.avi"))    #獲取影片
    new_video = clip.set_audio(audioclip)   #影片合併音頻
    new_video.write_videofile( os.path.join(r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama\Beelzebub-jou no Okinimesu mam.mp4")) 

if __name__ == "__main__" :
    main2()