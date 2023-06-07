import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Helper_private import shorten_number
from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks


def cut_point(var_hist,start_point,end_point,min_interval_set=1,max_interval_set=5):
    cut_point = []
    rec_state = False
    tmp_hist = []
    for i in range(start_point,end_point) :
        if rec_state : tmp_hist.append(var_hist[i])
        if not rec_state and not ( var_hist[i] < 0.8 ) :
            rec_state = True
            start_frame = i
            continue
        if rec_state and ( ( var_hist[i] < 0.8 ) or i == end_point-1) :
            rec_state = False
            end_frame = i
            if (( end_frame - 5 ) - ( start_frame + 5 ) > min_interval_set):
                #if not ( np.mix(tmp_hist[5:-5]) > 0.9 ) :
                middle_point = (start_frame + end_frame)/2
                left_point = max(start_frame+5,int( middle_point - max_interval_set / 2 ))
                right_point = min(end_frame-5,int( middle_point + max_interval_set / 2 ))
                if len(var_hist[left_point:right_point]) != 0 :
                    if not ( np.mean(np.array(var_hist[left_point:right_point]) >0.95) > 0.7 ) :
                    #mode = stats.mode(np.array(tmp_hist[5:-5]),axis=None, keepdims=True)[0]
                    #if not mode[0] > 0.9  :
                        cut_point.append( (left_point,right_point) )
                        #cut_point.append( (start_frame+5,end_frame-5) )
            tmp_hist = []
    return cut_point

def main():

    data_path = r"F:\work\video_analyze\my_work\video\data"
    data_list = os.listdir(data_path)

    output_path = r"F:\work\video_analyze\my_work\video\data\pic"
    for i in tqdm(data_list) :
        if ".txt" in i :
            pic_name = i.replace("_hist_var.txt","")
            var_hist = []

            with open(os.path.join(data_path,i),"r") as txtfile :
                lines = txtfile.readlines()
                for i in lines :
                    var_hist.append(eval(i))
            
            #print(var_hist)


            peaks, _ = find_peaks(1-np.array(var_hist), distance=30 , height=0.15)
            peaks_2, properties = find_peaks(1-np.array(var_hist), prominence=(None, 0.1))


            fake_hist = []
            for i in range(len( var_hist )) :
                if (i in peaks) and (i not in peaks_2):
                    fake_hist.append(1)
                else :
                    fake_hist.append(0)

            point_path = r"F:\work\video_analyze\my_work\video\data\point"
            with open(os.path.join(point_path,"{}_point.txt".format(pic_name)),"w") as txtfile:
                for i in range(len(fake_hist)) :
                    print(fake_hist[i],file=txtfile)

            """
            x_bar = np.arange( 0 , len( var_hist ) , 1 )
            fig = plt.figure(figsize=(50,8))
            plt.subplot(111)
            plt.plot( x_bar , var_hist , c = "r" )
            #plt.plot( peaks_2 , np.array(var_hist)[peaks_2] , "x", c = "y")
            #plt.plot( peaks , np.array(var_hist)[peaks] , "x", c = "b")
            plt.plot( x_bar , fake_hist , c="b" )
            plt.savefig(os.path.join(output_path,"{}.png".format(pic_name)),bbox_inches='tight',pad_inches = 0)
            plt.close('all')
            """

if __name__ == "__main__" :
    main()