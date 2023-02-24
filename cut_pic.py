import cv2
import os

def main():
    video_folder = r"F:\work\video_analyze\output\cut_video\Beelzebub-jou no Okinimesu mama"
    output_path = r"F:\work\video_analyze\output\img\cutimg"
    for idx,ele in enumerate( os.listdir(video_folder)) :
        video_path = os.path.join( video_folder , ele )
        vidCap = cv2.VideoCapture(video_path)
        ret = vidCap.grab()
        if not ret : continue
        ret,image = vidCap.retrieve()
        cv2.imwrite( os.path.join(output_path,"%s.png"%(idx)) , image )

if __name__ == "__main__" :
    main()