import cv2
import os
from tqdm import tqdm
import time
def main():

    data_path = r"output\var_hist"
    data_list = os.listdir(data_path)

    video_path = r"F:\work\video_analyze\data\video\Beelzebub-jou no Okinimesu mama"

    cut_video_path = r"F:\work\video_analyze\output\cut_video\Beelzebub-jou no Okinimesu mama"

    fps = 24
    sec_set = 5
    interval_set = fps*sec_set


    for i in tqdm(data_list) :
        pic_name = i.replace("_hist_var.txt","")
        var_hist = []

        with open(os.path.join(data_path,i),"r") as txtfile :
            lines = txtfile.readlines()
            for i in lines :
                var_hist.append(eval(i))
    
                        
        cut_point = []
        rec_state = False
        tmp_hist = []
        for i in range(len(var_hist)) :
            if rec_state : tmp_hist.append(var_hist[i])
            if not rec_state and not ( var_hist[i] < 0.8 ) :
                rec_state = True
                start_frame = i
                continue
            if rec_state and ( var_hist[i] < 0.8 ) :
                rec_state = False
                end_frame = i
                if (( end_frame - 5 ) - ( start_frame + 5 ) > interval_set) : 
                    if not ( min(tmp_hist[5:-5]) > 0.95 ) :
                        cut_point.append( (start_frame+5,end_frame-5) )
                tmp_hist = []

        
        vidCap = cv2.VideoCapture(os.path.join(video_path,"%s.mp4"%(pic_name)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width  = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = vidCap.get(cv2.CAP_PROP_FPS)

        cut_point_idx = 0
        #out = cv2.VideoWriter(os.path.join(cut_video_path,"%s_%s.mp4"%(pic_name,)),fourcc, FPS, (int(frame_width),int(frame_height)))
        frame_num = 0

        cut_state = False
        while True :
            ret = vidCap.grab()
            if not ret or cut_point_idx == len(cut_point): break
            if frame_num >=  cut_point[cut_point_idx][0] :
                start_time = time.time()
                if not cut_state :
                    cut_state = True
                    out = cv2.VideoWriter(os.path.join(cut_video_path,"%s_%s.mp4"%(pic_name,cut_point_idx)),fourcc, FPS, (frame_width,frame_height) )
                ret,image = vidCap.retrieve()
                out.write(image)
                if frame_num == cut_point[cut_point_idx][1] and cut_state :
                    cut_state = False
                    out.release()
                    cut_point_idx += 1
                end_time = time.time()
                print(" {} process : {} --- {:.2f} it/s".format(cut_point_idx,frame_num,1/(end_time-start_time)),end="\r")
            frame_num += 1
           

        vidCap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()