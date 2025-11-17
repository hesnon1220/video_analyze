import numpy as np
# Feature Scaling

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten, LSTM, TimeDistributed, RepeatVector, Embedding
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tensorflow.keras.optimizers import Adam
import yaml
from sklearn.preprocessing import StandardScaler
import time

log_dir = r"F:\work\video_analyze\my_work\output\log"
output_path = r"F:\work\video_analyze\my_work\output"
data_path = r"F:\work\video_analyze\my_work\train_data"
model_save_path = r"F:\work\video_analyze\my_work\output\model"
class_name = ['num','bass', 'drums', 'other', 'vocals']
fft_data = r"F:\work\video_analyze\my_work\sound\fft_data"
BIES_path = r"F:\work\video_analyze\my_work\sound\BIES_data"
data_set = r"F:\work\video_analyze\my_work\set_data"

"""
# 創建斷詞模型
def create_model(vocab_size, embedding_dim, hidden_units):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(hidden_units, return_sequences=True))  # 返回序列標記
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
"""

def buildManyToManyModel():
    model = Sequential()
    model.add(LSTM(units = 500 , input_shape=(500,735), return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(units = 250 , input_shape=(500,500) , return_sequences = True))
    #model.add(Dropout(0.2))
    model.add(Dense(units = 1, activation='softmax'))
    #model.compile(loss=tf.keras.losses.Huber(0.1), optimizer="adam")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def cut_y(input_list,target_list) :
    return_list = []

    tmp = []
    for idx,ele in enumerate(input_list) :
        tmp.append( target_list[idx] )
        if ele == 1 :
            return_list.append( tmp )
            tmp = []
    if tmp != [] :
        return_list.append( tmp )
    
    return return_list


def get_data(file_yml) :
    file_name = file_yml.replace(".yml","")
    print(file_name)
    with open(os.path.join(BIES_path,file_yml), 'r') as stream:
        y_data = yaml.load(stream,Loader=yaml.Loader)
    with open(os.path.join(fft_data,file_yml), 'r') as stream:
        x_data = yaml.load(stream,Loader=yaml.Loader)
    print(np.shape(x_data))
    return x_data,y_data

"""
def get_data(file_yml) :
    X_save = []

    file_name = file_yml.replace(".yml","")
    print(file_name)
    with open(os.path.join(data_path,file_yml), 'r') as stream:
        data = yaml.load(stream,Loader=yaml.Loader)
    
    min_len = min(len(data["point_data"]),len(data["vocals"]))
    scaler = StandardScaler()


    data["num"] = list(np.linspace(0,min_len,min_len+1))

    take_len = 30*10
    X_data = {}
    y_data = []




    for csn in class_name :
        X_data[csn] = []
        if csn is not "num" :
            scaler = scaler.fit(np.array(data[csn]).reshape(-1, 1))
            taget_data = scaler.transform(np.array(data[csn]).reshape(-1, 1))
        else : taget_data = np.array(data[csn]).reshape(-1, 1)
        for i in range( take_len , min_len ) :
            X_data[csn].append( [ i[0] for i in taget_data[ i-take_len : i]] )

    for i in range( take_len , min_len ) :
        #y_data.append( list(np.array(data["point_data"][ i-take_len : i])) )
        y_data.append(cut_y( data["point_data"][ i-take_len : i] , data["num"][ i-take_len : i] ))

    for i in range(len(y_data)) :
        tmp = []
        for j in range( 300 ) :
            tmp.append( [ X_data[csn][i][j] for csn in class_name] )
        X_save.append(tmp)

    return X_save,y_data
"""

def make_set_data() :

    max_len = 500

    total_len = []
    X_data = []
    Y_data = []

    for file_name in  tqdm(os.listdir(data_path)) :
        tmp_x,tmp_y =  get_data(file_name)
        if np.shape(tmp_x)[1] == 735 :
            X_data += tmp_x
            Y_data += tmp_y["point_data"]
            total_len += tmp_y["len_data"]
            print(np.shape(X_data))
    
    print(np.shape(X_data))
    scaler = StandardScaler()
    scaler = scaler.fit(X_data)
    #zero_data = scaler.transform(np.zeros([1,735]))

    tmp_len = 0
    tmp_cut = []
    cut_list = [] 

    while True :
        if len(total_len) == 0 :
            if len(tmp_cut) != 0 :
                cut_list.append( tmp_cut )
            break
        tmp_len += total_len[0]
        if tmp_len <= max_len : 
            tmp_cut.append( total_len.pop(0) )
        else :
            if len(tmp_cut) == 0 :
                cut_list.append( [total_len.pop(0)] )
            else :
                cut_list.append( tmp_cut )
                tmp_cut = []
            tmp_len = 0
        

    x_list = []
    y_list = []


    target_len_idx = 0
    tmp_fft = []
    tmp_BIES = []
    
    
    for idx,ele in tqdm(enumerate( Y_data )) :
        if ele == '[end]' : continue
        tmp_BIES.append(ele)
        tmp_fft.append( X_data[idx] )
        if len(tmp_BIES) == sum(cut_list[target_len_idx]) :
            for  _c_ in range(max_len - len(tmp_BIES)) :
                tmp_fft.append( [0]*735 )
                tmp_BIES.append('[end]')
            ds_dict = {
                "X" : scaler.transform(tmp_fft),
                "Y" : tmp_BIES
            }
            #x_list.append( scaler.transform(tmp_fft) )
            #y_list.append( tmp_BIES )
            tmp_fft = []
            tmp_BIES = []
            target_len_idx += 1
            with open(os.path.join(data_set,"{}.yml".format(target_len_idx)), 'w') as f:
                yaml.dump(ds_dict, f)


def change_BIES(input_list) :
    return_list = []
    tag = ["[end]","B","I","E"]
    for i in input_list :
        return_list.append( tag.index(i) )

    return return_list


def main() :
    X_data = []
    y_data = []
    for sd_n in tqdm(os.listdir( data_set )) :
        with open(os.path.join(data_set,sd_n), 'r') as stream:
            data = yaml.load(stream,Loader=yaml.Loader)
            if data["X"].shape != (500,735) : 
                print(sd_n)
                continue
            X_data.append( data["X"] )
            y_data.append( np.array(change_BIES( data["Y"] ) ))



    train_X = tf.convert_to_tensor(np.array(X_data))
    #train_Y = y_data
    train_Y = tf.convert_to_tensor(np.array(y_data))
    
    #print(train_Y)
    
    #X_train, X_test, y_train, y_test = train_test_split( train_X , train_Y , test_size=0.2 , random_state=42 )
    
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_freq=10, save_weights_only=True)
    model = buildManyToManyModel()
    model.fit(train_X, train_Y , epochs=500, batch_size=64,
            #validation_data=(X_test,y_test),validation_batch_size=64,
            callbacks=[callback,tensorboard_callback,model_checkpoint_callback])

    model.save(os.path.join( model_save_path , "{}.h5".format(time.strftime(r"%Y_%m_%d_%H_%M_%S", time.localtime()))))
    
    
def predict():
    model = models.load_model(r"F:\work\video_analyze\my_work\output\model\2023_06_07_03_26_38.h5")
    X_save,y_data = get_data("[乖離性ミリオンアーサー][Million Ways=One Destination][NEET].yml")
    X_data = np.array( X_save ).reshape( len(X_save),300,4 )
    
    result_list = []
    max_num = 0
    for i in tqdm(X_data):
        result =  model.predict(i.reshape(1,300,4))[0].reshape(1,-1)[0]
        result_list.append(result)
        if max(result) > max_num : max_num = max(result)
    print(max_num)
    

if __name__ == "__main__" :
    main()
    #predict()
    #get_data("[乖離性ミリオンアーサー][Million Ways=One Destination][NEET].yml")
    #print(X_save[0])
    #print(y_data[0])