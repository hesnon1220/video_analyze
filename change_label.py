import os
from tqdm import tqdm


path = r"F:\work\video_analyze\output\train_data_2\labels\train"

for i in tqdm(os.listdir(path)) :
    file_path = os.path.join(path,i)
    with open(file_path,"r") as txt_file :
        tmp = txt_file.readlines()
        return_data = []
        for k in tmp :
            return_i = k.replace("\n","").split()
            return_i[0] = str(int(return_i[0]) - 1)
            return_data.append(return_i)
    with open(file_path,"w") as txt_file :
        for r in return_data :
            print("\t".join(r),file = txt_file)