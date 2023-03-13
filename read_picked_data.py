import os
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import AudioFileClip,VideoFileClip
def get_interval_data(video_name,cut_idx) :
    #data_path = "F:\\work\\video_analyze\\output\\cut_video_data\\Detective Conan The Culprit Hanzawa\\%s.yaml"%(video_name)
    data_path = "F:\\work\\video_analyze\\output\\cut_video_data\\Beelzebub-jou no Okinimesu mama\\%s.yaml"%(video_name)
    with open( data_path,"r" ) as yamlfile :
        data_dict = yaml.load(yamlfile,Loader=yaml.Loader)
    return data_dict[video_name][cut_idx]["interval"]



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
    main()