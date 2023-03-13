import yaml
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from Helper_private import min_index,shorten_number,sigmoid
from tqdm import tqdm
import copy
import time

def main():
    #title = "Detective Conan The Culprit Hanzawa"
    title = "Beelzebub-jou no Okinimesu mama"
    save_path = "F:\\work\\video_analyze\\output\\cut_video_data\\%s"%(title)


    #base_path = r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa"
    base_path = r"F:\work\video_analyze\output\cut_video_data\Beelzebub-jou no Okinimesu mama"
    with open( os.path.join(base_path,"paragraph_dict.yaml"),"r" ) as yamlfile :
        paragraph_dict = yaml.load(yamlfile,Loader=yaml.Loader)

    video_data_list = [i.replace(".mp4",".yaml") for i in os.listdir(r"F:\work\video_analyze\data\video\Beelzebub-jou no Okinimesu mama") ]

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
            pick_score = (tmp_dict["Beelzebub"] >= 0 )*( tmp_dict["text"] == 0 )*( tmp_dict["title"] == 0 )*( min( gray_std ) >= 20 )*( min(gray_mean) >= 150 )#
            if pick_score : 
                write_data = [data_name,key,tmp_dict["lenght"]]
                all_video_data.append( write_data )


    print(all_video_data[:10])
    pick_data = []
    pick_data_dict = {}
    space_case_ori = {}
    for key,val in paragraph_dict["vocal"].items() :
        pick_data_dict[key] = {}
        space_case_ori[key] = []
        for idx,ele in enumerate(val) :
            pick_data_dict[key][idx] = []
            space_case_ori[key].append("")
            tmp_list = [(key,idx)]
            for avd_idx,avd_ele in enumerate(all_video_data) :
                if int(avd_ele[-1]*1.1) >= ele :
                    pick_data_dict[key][idx].append( avd_idx )
                    tmp_list.append( avd_idx )
            pick_data.append( tmp_list )



    ############################################################################################################
    pick_data_cp = copy.deepcopy(pick_data)
    space_lab = [-1]*len(pick_data_cp)
    xy_idx = [ i[0] for i in pick_data_cp ]
    while -1 in space_lab :
        cu_data_len = [ len(i)-1 for i in pick_data_cp ]
        target_idx = min_index(cu_data_len)[0]
        pop_data = pick_data_cp.pop(target_idx)
        sl_idx = xy_idx.index( pop_data[0] )
        if len(pop_data) > 1 :
            space_lab[sl_idx] = pop_data[1]
            for idx in range(len(pick_data_cp)) :
                if pop_data[1] in pick_data_cp[idx] :
                    pick_data_cp[idx].remove( pop_data[1])
        else :
            space_lab[sl_idx] = "None"
    if "None" in space_lab : raise Exception("太少")
    ############################################################################################################
    
    sorted_index = sorted(range(len(pick_data)), key = lambda k : len(pick_data[k]),reverse=False) 
    sorted_pick_data = [ pick_data[i] for i in sorted_index ]
    case_list = []
    case_num = 0
    h_list = [-1]
    while True :
        start_time = time.time()
        pick_data_tmp = copy.deepcopy(sorted_pick_data)
        rec_list = []
        q = case_num
        if_break = False
        for idx_pd in range( len(pick_data) ) :
            pd_lengh = [ len(i)-1 for i in  pick_data_tmp]
            if 0 in pd_lengh[idx_pd:] : 
                if_break = True
                break
            r = q // pd_lengh[idx_pd]
            q = q % pd_lengh[idx_pd]
            rec_num = pick_data_tmp[idx_pd][q+1]
            rec_list.append( rec_num )
            for i_tmp in range( len(pick_data) ) :
                if rec_num in pick_data_tmp[i_tmp] : pick_data_tmp[i_tmp].remove( rec_num )
            q = r
        end_time = time.time()
        case_num += 500
        #h_list.append( -np.log10(1-len(case_list)/(case_num+1)) )
        if if_break : continue
        try :   print( "{}/{}={}---{} it/s".format( len(case_list),(case_num+1),len(case_list)/(case_num+1),shorten_number(1/(end_time-start_time))),end="\r" )
        except : pass
        if q != 0 or len(case_list) <= h_list[-1] or case_num >= 1e7 or len(case_list) >= 1e6 : break
        h_list.append(len(case_list))
        if len( rec_list ) == len(pick_data) and rec_list not in case_list: case_list.append(rec_list)   


    min_score = -1
    max_score = -1
    chosen_case = {}
    for case_d in case_list :
        space_case = copy.deepcopy(space_case_ori)
        for idx,ele in enumerate(case_d) :
            x = sorted_pick_data[idx][0][0]
            y = sorted_pick_data[idx][0][1]
            #if x not in case_dict.keys() : case_dict[ x ] = 
            space_case[x][y] = ele

        socre_list = []
        for key,val in space_case.items() :
            rec_socre = 0     
            for target_num in range(1,len(val)) :
                f_data = all_video_data_dict[all_video_data[val[target_num-1]][0]][all_video_data[val[target_num-1]][1]]["BGR"]
                b_data = all_video_data_dict[all_video_data[val[target_num]][0]][all_video_data[val[target_num]][1]]["BGR"]
                max_len = min(paragraph_dict["vocal"][key][target_num-1],all_video_data[val[target_num-1]][2])
                for BGR_idx in range(3) :
                    rec_socre += RGB_cal(f_data[BGR_idx][int(max_len-10):max_len],b_data[BGR_idx][:10])
            socre_list.append( rec_socre )
        total_score = sum(socre_list)
        #if min_score > total_score or min_score == -1:
            #min_score = total_score
            #chosen_case = space_case
        if max_score < total_score or max_score == -1:
            max_score = total_score
            chosen_case = space_case

    #print(chosen_case)
    used_index = []
    for val in chosen_case.values() :
        used_index += val

    vocal_dict ={}
    for key,val in chosen_case.items() :
        vocal_dict[key] = []
        for ele in val :
            vocal_dict[key].append( all_video_data[ele] )

    print(vocal_dict)


    #for key,val in paragraph_dict["interlude"].items() :

    score_list = []
    interlude_data = []
    for avd_idx,avd_ele in tqdm(enumerate(all_video_data)) :
        if avd_idx not in used_index :
            RGB_data = all_video_data_dict[avd_ele[0]][avd_ele[1]]["BGR"]
            score = []
            for BGR_idx in range(3) :
                score.append(np.sum(np.abs(np.array(RGB_data[BGR_idx][:-1])-np.array(RGB_data[BGR_idx][1:]))))
            score_list.append( sum(score) )
            interlude_data.append( avd_ele )
    


                
    sorted_index_byscore = sorted(range(len(interlude_data)), key = lambda k : score_list[k],reverse=True)
    sorted_index_byscorelen = sorted(range(len(interlude_data)), key = lambda k : score_list[k]/interlude_data[k][-1],reverse=False)
    sorted_pick_data_byscore = [ interlude_data[i] for i in sorted_index_byscore ]
    sorted_pick_data_byscorelen = [ interlude_data[i] for i in sorted_index_byscorelen ]
    interlude_dict = {}
    for idx,ele in tqdm(enumerate(paragraph_dict["interlude"])) :
        interlude_dict[idx] = []
        #for idx_spd,ele_spd in enumerate(sorted_pick_data_byscore) :
            #if ele_spd[-1] >= ele :
                #interlude_dict[idx].append( ele_spd )
                #sorted_pick_data_byscore.remove(ele_spd)
                #break
        if len(interlude_dict[idx]) == 0 :
            
            tmp_rec = []
            for idx_spd,ele_spd in enumerate(sorted_pick_data_byscorelen) :
                tmp_rec.append( ele_spd )
                if int(sum([ i[-1] for i in tmp_rec])*1.1) > ele : break
            interlude_dict[idx] += tmp_rec
            for t_ele in tmp_rec :
                sorted_pick_data_byscorelen.remove(t_ele)



    paragraph_dict["vocal_data"] = vocal_dict
    paragraph_dict["interlude_data"] = interlude_dict

    with open(os.path.join(save_path,'picked_data.yaml'), 'w') as f:
        yaml.dump(paragraph_dict, f)



    """

    x_bar = np.arange( 0 , len( h_list ) , 1 )
    fig = plt.figure(figsize=(50,8))
    plt.subplot(111)
    plt.plot( x_bar , h_list , c = "r" )
    plt.savefig("h_list.png",bbox_inches='tight',pad_inches = 0)
    plt.close('all')


    case_num = 0
    while True :
        pick_data_tmp = pick_data.copy()
        rec_list = []
        q = case_num
        for idx_pd in range( len(pick_data) ) :
            pd_lengh = [ len(i)-1 for i in  pick_data_tmp]
            if 0 in pd_lengh : break
            r = q // pd_lengh[idx_pd]
            q = q // pd_lengh[idx_pd]
            rec_list.append( q )
            q = r
        if len( rec_list ) == len(pick_data) : print(rec_list)
        case_num += 1


    pick_data_save = pick_data.copy()
    space_lab = [-1]*len(pick_data)
    xy_idx = [ i[0] for i in pick_data ]

    while -1 in space_lab :
        print("="*10)
        cu_data_len = [ len(i)-1 for i in pick_data ]
        print(cu_data_len)
        target_idx = min_index(cu_data_len)[0]
        pop_data = pick_data.pop(target_idx)
        sl_idx = xy_idx.index( pop_data[0] )
        if len(pop_data) > 1 :
            space_lab[sl_idx] = pop_data[1]
            for idx in range(len(pick_data)) :
                if pop_data[1] in pick_data[idx] :
                    pick_data[idx].remove( pop_data[1])
        else :
            space_lab[sl_idx] = "None"
        print(space_lab)
    

    if "None" in space_lab : raise Exception("太少")

    print(pick_data_save)
    print(space_lab)
    


    for i in range(len( space_lab )) :
        (key_1,key_2) = pick_data_save[i][0]
        for j in range(len( space_lab )) :
            if j != i :
                if space_lab[j] in pick_data_dict[key_1][key_2] :
                    pick_data_dict[key_1][key_2].remove( space_lab[j] )
    print(pick_data_dict)
    

    list_pick_data = []
    for first_key,first_val in pick_data_dict.items() :
        for second_key,second_val in first_val.items() :
            list_pick_data.append(second_val)
    


   
            if len(interlude_dict[idx]) == 0 :
            tmp_rec = []
            for idx_spd,ele_spd in enumerate(sorted_pick_data) :
                tmp_rec.append(  )




    used_index = []
    result_data = {}
    for key,val_dict in tqdm(pick_data_dict.items()) :
        result_data[key] = []
        rec_lsit_list = []
        score_list = []
        for s_idx in range(len(val_dict[0])) :
            f_idx =  val_dict[0][s_idx]
            if f_idx in used_index : continue
            rec_list = [f_idx]
            rec_socre = 0
            for step_num in range(1,len(val_dict.keys())) :
                b_ele_list = []
                tmp_score = []
                for b_idx in val_dict[step_num] :
                    if b_idx in rec_list or b_idx in used_index: continue
                    
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
    
    """
    






def RGB_cal(f_RGB,b_RGB):
    return  np.sum(np.power(np.array(f_RGB) - np.array(b_RGB),2))


if __name__ == "__main__" :
    main()