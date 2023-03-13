import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def main() :

    pick_data = [[1,2,3,4,7],[1,2,3,5,6],[1,2,5,7],[3,5,7]]


    sorted_index = sorted(range(len(pick_data)), key = lambda k : len(pick_data[k]),reverse=False) 

    sorted_pick_data = [ pick_data[i] for i in sorted_index ]

    case_list = []
    case_num = 0
    h_list = []
    while case_num<1000 :
        pick_data_tmp = copy.deepcopy(sorted_pick_data)
        rec_list = []
        q = case_num
        if_break = False
        q_list = []
        for idx_pd in range( len(pick_data) ) :
            pd_lengh = [ len(i) for i in  pick_data_tmp]
            if 0 in pd_lengh[idx_pd:] : 
                if_break = True
                break
            r = q // pd_lengh[idx_pd]
            q = q % pd_lengh[idx_pd]
            rec_num = pick_data_tmp[idx_pd][q]
            q_list.append(q)
            rec_list.append( rec_num )
            for i_tmp in range( len(pick_data) ) :
                if rec_num in pick_data_tmp[i_tmp] : pick_data_tmp[i_tmp].remove( rec_num )
            q = r
        case_num += 1
        h_list.append( len(case_list)/(case_num+1) )
        print( "{}/{}={}".format(len(case_list), case_num,len(case_list)/(case_num+1)) )
        if if_break : continue
        if len( rec_list ) == len(pick_data) and rec_list not in case_list:
            #print(case_num,q,rec_list)
            case_list.append(rec_list)
            
        
    print("*"*10)
    print(sorted_pick_data)
    print(case_list)
    print(  [7,5,4,6] in case_list )
    
    x_bar = np.arange( 0 , len( h_list ) , 1 )
    fig = plt.figure(figsize=(50,8))
    plt.subplot(111)
    plt.plot( x_bar , h_list , c = "r" )
    plt.savefig("h_list.png",bbox_inches='tight',pad_inches = 0)
    plt.close('all')


if __name__ == "__main__" :
    main()