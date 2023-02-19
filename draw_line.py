import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Helper_private import shorten_number


def main():

    data_path = r"output"
    data_list = os.listdir(data_path)

    output_path = r"img"
    for i in tqdm(data_list) :
        pic_name = i.replace("txt","")
        var_hist = []

        with open(os.path.join(data_path,i),"r") as txtfile :
            lines = txtfile.readlines()
            for i in lines :
                var_hist.append(eval(i))
        
        #print(var_hist)

        x_bar = np.arange( 0 , len( var_hist ) , 1 )
        fig = plt.figure(figsize=(20,8))
        plt.subplot(111)
        plt.plot( x_bar , var_hist , c = "r" )
        plt.savefig(os.path.join(output_path,"{}.png".format(pic_name)),bbox_inches='tight',pad_inches = 0)
        plt.close('all')

if __name__ == "__main__" :
    main()