import yaml
import os
import numpy as np
from Helper_private import min_index,shorten_number

def main():
    base_path = r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa"
    with open( os.path.join(base_path,"paragraph_dict.yaml"),"r" ) as yamlfile :
        paragraph_dict = yaml.load(yamlfile,Loader=yaml.Loader)

    video_data_list = [i.replace(".mp4",".yaml") for i in os.listdir(r"F:\work\video_analyze\data\video\Detective Conan The Culprit Hanzawa") ]

    print(paragraph_dict)
    print(video_data_list)
    
    all_video_data_dict = {}


    all_video_data = []
    for video_data_name in video_data_list :
        data_path = os.path.join(base_path,video_data_name)
        data_name = video_data_name.replace(".yaml","")
        with open( data_path,"r" ) as yamlfile :
            video_data_dict = yaml.load(yamlfile,Loader=yaml.Loader)
        all_video_data_dict[data_name] = video_data_dict[data_name]
        print(video_data_dict[data_name].keys())
        for key in list(video_data_dict[data_name].keys()) :
            tmp_dict = video_data_dict[data_name][key]
            gray_mean = tmp_dict["gray_mean"]
            gray_std = tmp_dict["gray_std"]
            pick_score = (tmp_dict["black"] >= 0 )*( tmp_dict["text"] == 0 )*( tmp_dict["title"] == 0 )*( min( gray_std ) >= 15 )#*( min(gray_mean) >= 100 )#
            if pick_score : 
                write_data = [data_name,key,tmp_dict["lenght"]]
                all_video_data.append( write_data )

    pick_data = {}
    for key,val in paragraph_dict["vocal"].items() :
        pick_data[key] = {}
        for idx,ele in enumerate(val) :
            pick_data[key][idx] = []
            for avd_idx,avd_ele in enumerate(all_video_data) :
                if avd_ele[-1] >= ele :
                    pick_data[key][idx].append( avd_idx )

    used_index = []
    result_data = {}
    for key,val_dict in pick_data.items() :
        result_data[key] = []
        rec_lsit_list = []
        score_list = []
        for s_idx in range(len(val_dict[0])) :
            f_idx =  val_dict[0][s_idx]
            rec_list = [f_idx]
            rec_socre = 0
            for step_num in range(1,len(val_dict.keys())) :
                b_ele_list = []
                tmp_score = []
                print(val_dict[step_num])
                for b_idx in val_dict[step_num] :
                    
                    if b_idx in rec_list : continue
                    
                    
                    b_ele_list.append(b_idx)

                    f_data = all_video_data[f_idx]
                    f_data = all_video_data_dict[all_video_data[f_idx][0]][all_video_data[f_idx][1]]["BGR"]
                    b_data = all_video_data_dict[all_video_data[b_idx][0]][all_video_data[b_idx][1]]["BGR"]
                    
                    max_len = paragraph_dict["vocal"][key][step_num-1]
                    BGR_score = 0
                    for BGR_idx in range(3) :
                        BGR_score += RGB_cal(f_data[BGR_idx][int(max_len-10):max_len],b_data[BGR_idx][:10])
                    tmp_score.append(BGR_score)
                if len(tmp_score) == 0 : continue
                rec_socre += min(tmp_score)
                rec_list.append(b_ele_list[min_index(tmp_score)[0]])
                f_idx = rec_list[-1]
            rec_lsit_list.append(rec_list)
            score_list.append( rec_socre )
        result_data[key] = rec_lsit_list[ min_index(score_list)[0] ]  
        used_index += [ i for i in result_data[key] ]
        print(result_data)

def RGB_cal(f_RGB,b_RGB):
    return  shorten_number(np.power(np.sum(np.power(np.array(f_RGB) - np.array(b_RGB),2)),1/2))


if __name__ == "__main__" :
    main()