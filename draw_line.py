import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Helper import shorten_number




def main():


    var_hist = []

    with open(r"output\hist_var.txt","r") as txtfile :
        lines = txtfile.readlines()
        for i in lines :
            var_hist.append(eval(i))
    
    print(var_hist)

    x_bar = np.arange( 0 , len( var_hist ) , 1 )
    fig = plt.figure()
    plt.subplot(111)
    plt.plot( x_bar , var_hist , c = "r" )
    plt.savefig(os.path.join(r"output\var_hist.png"),bbox_inches='tight',pad_inches = 0)
    plt.close('all')

if __name__ == "__main__" :
    main()