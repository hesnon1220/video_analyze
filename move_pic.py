import shutil
import os

img_path = r"F:\work\video_analyze\output\train_data_2\images\no_train"
label_path = r"F:\work\video_analyze\output\train_data_2\labels\train"
target_path = r"F:\work\video_analyze\output\train_data_2\images\train"


for i in os.listdir(label_path) :
    pic_name = i.replace(".txt",".png")
    shutil.copy( os.path.join( img_path , pic_name ) , os.path.join( target_path , pic_name ) )