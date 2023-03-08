import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import math
from skimage.segmentation import clear_border
from skimage import morphology
from skimage.metrics import structural_similarity
from tqdm import tqdm
from moviepy.editor import AudioFileClip,VideoFileClip
import scipy.signal
from scipy.io import wavfile
####################################################################################################################################
def area_cal(pig_p , p_list:list) :
    '''
    重疊面積計算
    pig_p:主要面積
    p_list:次要面積list
    '''
    area_coord = []
    area_per = []
    for i in p_list :
        p1 = ( i[0] , i[1] )
        p2 = ( i[0] , i[3] )
        p3 = ( i[2] , i[3] )
        p4 = ( i[2] , i[1] )
        detect = False
        for p in [p1,p2,p3,p4] :
            if ( p[0] >= pig_p[0] and p[0] <= pig_p[2] ) and ( p[1] >= pig_p[1] and i[1] <= pig_p[3] ) :

                x0 = max( i[0] , pig_p[0] ) 
                x1 = min( i[2] , pig_p[2] )
                y0 = max( i[1] , pig_p[1] )
                y1 = min( i[3] , pig_p[3] )

                piglet_area = ( i[2] - i[0] ) * ( i[3] - i[1] )
                pp_area =  ( x1 - x0 ) * ( y1 - y0 )
                area_coord.append( [ x0 , y0 , x1 , y1 ] )
                area_per.append( pp_area / max(piglet_area,1) )
                detect = True
                break
        if not detect :
            area_coord.append([])
            area_per.append(0)
    return area_coord , area_per
####################################################################################################################################
def dataframe_change(data_frame_list):
    '''
    轉換dataframe
    '''
    return_list = []
    for data_frame in data_frame_list :
        tmp = []
        if not data_frame.empty :
            data_array = np.array(data_frame)
            data_lengh , _ = np.shape(data_array)
            for i in range(data_lengh) :
                tmp.append(list(data_array[i][:6]))
        return_list.append(tmp.copy())
    return return_list
####################################################################################################################################
def coord_change( coord , width , hight ) :
    '''
    座標轉換
    [cx,cy,rw,rh]->[cx,cy,x0,y0,x1,y1]
    '''
    cx = int(coord[0]*width)
    cy = int(coord[1]*hight)
    rw = coord[2]*width
    rh = coord[3]*hight

    x0 = int(cx - rw/2)
    y0 = int(cy - rh/2)
    x1 = int(cx + rw/2)
    y1 = int(cy + rh/2)
    return [cx,cy,x0,y0,x1,y1]
####################################################################################################################################
def wh_cal(coord) :
    '''
    矩形寬高計算
    [x0,y0,x1,y1]->[w,h]
    '''
    x0 = coord[0]
    y0 = coord[1]
    x1 = coord[2]
    y1 = coord[3]
    w = y1-y0
    h = x1-x0
    return w,h
####################################################################################################################################
def uncoord_change(coord, width , hight ) :
    '''
    座標反向轉換
    [cx,cy,x0,y0,x1,y1]->[cx,cy,rw,rh]
    '''
    x0 = coord[0]
    y0 = coord[1]
    x1 = coord[2]
    y1 = coord[3]

    cx = ( x0 + x1 ) / 2 / width
    cy = ( y0 + y1 ) / 2 / hight
    rw = (x1 - x0) / width
    rh = (y1 - y0) / hight
    return [cx,cy,rw,rh]
####################################################################################################################################
def sigmoid(x,alpha=0,beta=1) :
    '''
    S型函數
    '''
    y = 1 / (1 + np.exp(-(x-alpha)*beta))
    return y
####################################################################################################################################
def threshold_color(value,threshold,color_list = [(0,0,255),(255,0,0)],txt_list = ["",""]) :
    '''
    色彩、文字閥值
    '''
    if value >= threshold : return color_list[0] , txt_list[0]
    else : return color_list[1] , txt_list[1]
####################################################################################################################################
def feed_cal(pig_center,p_list , area_per,pig_posture=2):
    '''
    哺乳分數計算
    pig_center:母豬中心
    p_list:小豬位置列表
    area_per:面積占比
    pig_posture:母豬姿態
    '''
    score_list= []
    for _idx_ in range(len(area_per)) :
        if pig_posture != 2 : score = 0
        else :
            ##############################################################################################
            p_coord = p_list[_idx_]
            vector_x = p_coord[0]-p_coord[2]
            vector_y = p_coord[1]-p_coord[3]
            if vector_y != 0 :
                p = vector_x / vector_y
                pv = 1 / (1 + p*p)
            else :
                pv = 0
            ##############################################################################################
            vector_posx = ( p_coord[0] + p_coord[2] ) / 2 - pig_center[0]
            vector_posy = ( p_coord[1] + p_coord[3] ) / 2 - pig_center[1]
            if vector_posy != 0 :
                p = vector_posx / vector_posy
                pos = 1 / (1 + p*p)
            else :
                pos = 0
            ##############################################################################################
            score_weights = [2,1,4] #重疊面積、仔豬面向、母豬仔豬夾角
            score = np.sign(vector_posy) * pow(( pow(sigmoid(area_per[_idx_],0.3,10),score_weights[0])*pow(sigmoid(pv,0.5,10),score_weights[1])*pow(sigmoid( pos ,0.3,10),score_weights[2]) ),1 / np.sum(score_weights))
        score_list.append(score)
    return score_list
####################################################################################################################################
def shorten_number(num:float,lenght:int=4)->float:
    '''
    取小數點後n位
    num:原數值
    lenght:取值長度
    '''
    try: return_num = math.floor(num*pow(10,lenght)+0.5)/pow(10,lenght)
    except : raise Exception( "num  : %s ; Lenght : %s"%(num,lenght) )
    return  return_num
####################################################################################################################################
def fixed_area_detect(image,threshold_color = 127.5) :
    '''
    固定區域黑白閥值
    '''
    gray_image = cv2.cvtColor( image , cv2.COLOR_BGR2GRAY )
    hist = [ i[0] for i in cv2.calcHist([gray_image],[0],None,[256],[0.0,255.0]) ]
    color = np.linspace(0,len(hist)-1,len(hist))-threshold_color
    tmp = []
    c = threshold_color/256
    for i in range(int(len(color)*c)) :
        tmp.append(color[i] * hist[i])
    w,h = gray_image.shape
    return 1+(np.sum(tmp) / int(len(color)/2) / (w*h))
####################################################################################################################################
def draw_distribution(input_list,save_path,save_name,threshold=None,y_lim = None):
    '''
    繪製分布圖
    '''
    x_bar = np.arange( 0 , len( input_list ) , 1 )
    fig = plt.figure()
    plt.subplot(111)
    plt.plot( x_bar , input_list , c = "r" )
    if threshold != None :
        #plt.plot( x_bar , np.ones(len(input_list))*threshold , c = 'g' )
        actiivate_f = np.array(input_list) >= threshold
        #plt.plot( x_bar , actiivate_f, c = 'b' )
        _idx_ = 0
        sub_bar = []
        while _idx_ < len(actiivate_f) :
            if actiivate_f[_idx_] == True :
                sub_bar.append(x_bar[_idx_])
                if _idx_+1 == len(actiivate_f) : plt.fill_between(sub_bar, 0, 1, color='b',alpha =0.3)
            elif actiivate_f[_idx_] == False:
                if sub_bar != [] :
                    plt.fill_between(sub_bar, 0, 1, color='b',alpha =0.3)
                    sub_bar = []
            _idx_ += 1
    plt.xlim(0,len( input_list ))
    if y_lim != None :
        plt.ylim(y_lim[0],y_lim[1])
    plt.xlabel( "sec" )
    plt.ylabel( "distribution" )
    plt.title(save_name)
    plt.savefig(os.path.join(save_path,"{}.png".format(save_name)),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
    return
####################################################################################################################################
def draw_Compare_distribution(list_1,list_2,save_path,title,save_name,):
    '''
    繪製比較分布圖
    '''
    x_bar = np.arange( 0 , len( list_1 ) , 1 )
    if len(list_1) != len(list_2) : raise Exception( "lenght error : %s != %s "%( len(list_1) , len(list_2) ) )
    fig = plt.figure(figsize=(12,3))
    plt.subplot(111)
    plt.plot( x_bar , [0]*len(x_bar) , c = "r" )
    data_list = [list_1,list_2]
    setting_list = [(0,1,'b',0.3),(0,-1,'g',0.3)]

    for id in range(2) :
        _idx_ = 0
        sub_bar = []
        while _idx_ < len(data_list[id]) :
            if data_list[id][_idx_] == True :
                sub_bar.append(x_bar[_idx_])
                if _idx_+1 == len(data_list[id]) : plt.fill_between(sub_bar, setting_list[id][0], setting_list[id][1], color=setting_list[id][2],alpha=setting_list[id][3])
            elif data_list[id][_idx_] == False:
                if sub_bar != [] :
                    plt.fill_between(sub_bar, setting_list[id][0], setting_list[id][1], color=setting_list[id][2],alpha=setting_list[id][3])
                    sub_bar = []
            _idx_ += 1
    plt.xlim(0,len( list_1 ))
    plt.ylim(setting_list[1][1],setting_list[0][1])
    plt.xlabel( "sec" )
    plt.title(title)
    plt.savefig(os.path.join(save_path,"{}.png".format(save_name)),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
    return
####################################################################################################################################
def draw_scatter( input_list,save_path,save_name ) :
    '''
    繪製點狀分布
    '''
    fig = plt.figure()
    plt.subplot(111)
    x_bar = np.arange( 0 , len( input_list ) , 1 )
    plt.scatter( x_bar , input_list , color = 'r' )
    plt.title(save_name)
    plt.savefig(os.path.join(save_path,"{}.png".format(save_name)))
    plt.close('all')
    return
####################################################################################################################################
def calculate_time(input_list,threshold=0.5) :
    '''
    計算時間、總時間、資料長度
    '''
    actiivate_f = np.array(input_list) >= threshold
    total_time = sum(actiivate_f)
    data_length = len(input_list)
    return  actiivate_f,total_time,data_length
####################################################################################################################################
def time_gate_trans(input_time):
    '''
    時間閥值轉換
    小時H、分鐘M、秒S
    '''
    H = int(input_time.strftime("%H"))
    M = int(input_time.strftime("%M"))
    S = int(input_time.strftime("%S"))
    return H
####################################################################################################################################
def hr_activity(hrga_list,h,w):
    '''
    網格行動力計算
    '''
    grid_array = np.zeros((h,w))
    for grid in hrga_list :
        for i in grid[1:] :
            grid_array[i[0],i[1]] += 1
    grid_array = grid_array.reshape(np.size(grid_array))
    k = np.array(grid_array>0)
    area_persent = np.sum(k)/len(k)
    tmp = []
    for i in grid_array :
        if i > 0 : tmp.append(i)
    area_statistics = (np.max(tmp),np.min(tmp),np.mean(tmp),np.std(tmp)) if len(tmp) != 0 else (0,0,0,0)
    return area_persent,area_statistics
####################################################################################################################################
def piglet_moves(data_list):
    '''
    移動格數計算
    '''
    tmp = []
    for coord in data_list:
        for c in coord :
            if c not in tmp :
                tmp.append(c)
    return len(tmp)
####################################################################################################################################
def hr2fr(h,m,s,fr=10):
    '''
    計算總幀數
    '''
    return (60*60*h+60*m+s)*fr
####################################################################################################################################
def distance_cal(point_1:tuple,point_2:tuple)->float:
    '''
    距離計算
    '''
    x0 = point_1[0]
    y0 = point_1[1]
    x1 = point_2[0]
    y1 = point_2[1]

    distance = shorten_number( pow( pow( x0 - x1 , 2 ) + pow( y0 - y1 , 2 ) , 1/2 ) )
    return distance
####################################################################################################################################
def area_point_detect(area:tuple,point:tuple)->bool :
    '''
    點與區域是否重疊
    '''
    xmin = area[0]
    xmax = area[2]
    ymin = area[1]
    ymax = area[3]

    x_o = point[0]
    y_o = point[1]

    if xmin <= x_o and x_o <= xmax and ymin <= y_o and y_o <= ymax : return True
    else : return False
####################################################################################################################################
def iou_cal(area_1,area_2)->float:
    '''
    iou計算
    '''
    a_x1 = area_1[0]
    a_x2 = area_1[2]
    a_y1 = area_1[1]
    a_y2 = area_1[3]

    b_x1 = area_2[0]
    b_x2 = area_2[2]
    b_y1 = area_2[1]
    b_y2 = area_2[3]

    ab_max_x1 = min(a_x1,b_x1)
    ab_max_x2 = max(a_x2,b_x2)
    ab_max_y1 = min(a_y1,b_y1)
    ab_max_y2 = max(a_y2,b_y2)


    if_inarea = False
    for point in [  (b_x1,b_y1),(b_x2,b_y1),(b_x1,b_y2),(b_x2,b_y2) ] :
        if_inarea += area_point_detect(area_1,point)
    if not if_inarea : ab_cross_area = 0
    else : 
        ab_cross_x1 = max(a_x1,b_x1)
        ab_cross_x2 = min(a_x2,b_x2)
        ab_cross_y1 = max(a_y1,b_y1)
        ab_cross_y2 = min(a_y2,b_y2)
        ab_cross_area = (ab_cross_x2-ab_cross_x1)*(ab_cross_y2-ab_cross_y1)

    a_area = (a_x2-a_x1)*(a_y2-a_y1)
    b_area = (b_x2-b_x1)*(b_y2-b_y1)
    ab_max_area = (ab_max_x2-ab_max_x1)*(ab_max_y2-ab_max_y1)
    if ab_max_area != 0 :   return shorten_number( ab_cross_area/ab_max_area )
    else : return 0
####################################################################################################################################
def moves_cal(moves_list) :
    '''
    移動力計算
    '''
    cal_tmp = []
    for i in moves_list :
        if i != () :
            cal_tmp.append( i )
    
    if len(cal_tmp) >= 2 :
        moves = 0
        for i in range(len(cal_tmp)-1) :
            start_point = cal_tmp[i]
            end_point = cal_tmp[i+1]
            moves += distance_cal( start_point , end_point )
        return shorten_number( moves/10 , lenght=2 )
    else : return 0
####################################################################################################################################
def renew_list(data_list:list,input_data)->list :
    '''
    向後更新列表
    '''
    data_list[:-1] = data_list[1:]
    data_list[-1] = input_data
    return data_list
####################################################################################################################################
def find_most_value(input_list:list):
    '''
    尋找出現最多次的值
    '''
    return max(set(input_list),key=input_list.count)
####################################################################################################################################
def make_threshold_img(image):
    '''
    繪製二值化圖像
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (9, 9), 0)
    kernel = np.ones((9, 9), np.uint8)
    open = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel, iterations=3)
    ret, open_th = cv2.threshold(open,80,255,cv2.THRESH_BINARY)
    cleared = clear_border(open_th)
    if np.max(cleared) != 255 : return cleared
    else : 
        chull = morphology.convex_hull_image(cleared)
        return np.array(chull*255,dtype="uint8")
####################################################################################################################################
def make_threshold_img_mask(image):
    '''
    繪製二值化圖像
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_pixel = np.array([0,10,80],dtype="uint8")
    upper_pixel = np.array([200,255,255],dtype="uint8")
    mask = cv2.inRange(hsv,lower_pixel,upper_pixel)
    cleared = clear_border(mask)
    if np.max(cleared) != 255 : return cleared
    else : 
        chull = morphology.convex_hull_image(cleared)
        return np.array(chull*255,dtype="uint8")
####################################################################################################################################
def get_hist(img,channel=0) :
    hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX, -1)
    return hist
####################################################################################################################################
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    tmp = 0
    for l,r in zip(lh, rh) :
        r = max(0,r)
        l = max(0,l)
        tmp += 1 - (0 if (l == 0 and  r == 0) else float(abs(l - r)) / max(l, r))
    hist = tmp / len(lh)
    return shorten_number( hist[0] )
####################################################################################################################################
def get_list_barycenter(data_list) :
    sum = 0
    for idx,ele in enumerate(data_list) :
        sum += idx*ele
    return shorten_number(sum)
####################################################################################################################################
# 差异值哈希算法
def dhash(image):
    # 将图片转化为8*8
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result
####################################################################################################################################
def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n
####################################################################################################################################
def skss(standard_gray,compare_gray):
    return structural_similarity(standard_gray,compare_gray,data_range=compare_gray.max() - compare_gray.min())
####################################################################################################################################
def fft(input_data,samplerate,sampling_interval) -> list:
    lenth = np.shape(input_data)[0] #資料長度
    d1 = np.array([ input_data[i][0] for i in range(lenth) ])   #單聲道資料
    fft_interval = int(samplerate * sampling_interval)  #計算區間
    result = []
    for i in tqdm(range(int(lenth/fft_interval))):    #分段進行傅立葉轉換
        fft_tmp = np.abs(np.fft.fft(d1[fft_interval*i :fft_interval*(i+1)]))
        fft_tmp = fft_tmp[:int(np.shape(fft_tmp)[0]/2) ]    #保留一半傅立葉轉換
        result.append(fft_tmp)
    return result
####################################################################################################################################
def mp32wav(root,extension)->str:   #轉檔
    audioclip = AudioFileClip(root + extension) #獲取音頻位置
    audioclip.write_audiofile(root + ".wav")    #轉換為wav
    return ( root + ".wav" )    #回傳新檔名
####################################################################################################################################
def draw_single_lines(input_data,output_path,save_name):
    x_bar = np.arange( 0 , len( input_data ) , 1 )
    #peaks = scipy.signal.find_peaks_cwt(input_data,5)
    peaks,_ = scipy.signal.find_peaks(input_data)
    fig = plt.figure(figsize=(20,8))
    plt.subplot(111)
    plt.plot( x_bar , input_data , c = "r" )
    plt.scatter(peaks, input_data[peaks])
    plt.savefig(os.path.join(output_path,"{}.png".format(save_name)),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
####################################################################################################################################
def get_fft(file_path,fps):
    sampling_interval = 1/fps   #間隔
    root, extension = os.path.splitext(file_path)    #獲取檔案副檔名
    if extension != ".wav" :    #轉檔wav
        file_path = mp32wav(root,extension)
    samplerate, data = wavfile.read(file_path)   #獲取採樣率、振幅(雙聲道)
    #print(samplerate)
    #print(data.shape)
    result = fft(data,samplerate,sampling_interval) #傅立葉轉換
    return result
####################################################################################################################################
def single_distance(input_data):
    return shorten_number( np.power(np.sum([ i*i for i in input_data ]),1/2) )


if __name__ =="__main__" :
    print("This is Helper_private.")