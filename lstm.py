import numpy as np
# Feature Scaling

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten, LSTM, TimeDistributed, RepeatVector
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
class_name = ['bass', 'drums', 'other', 'vocals']


def buildManyToManyModel():
    model = Sequential()
    model.add(LSTM(units = 100 , input_shape=(300,4), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50 , input_shape=(300,100) , return_sequences = True))
    model.add(LSTM(units = 25 , input_shape=(300,50) , return_sequences = True))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(loss=tf.keras.losses.Huber(0.01), optimizer="adam")
    model.summary()
    return model


def get_data(file_yml) :
    X_save = []

    file_name = file_yml.replace(".yml","")
    print(file_name)
    with open(os.path.join(data_path,file_yml), 'r') as stream:
        data = yaml.load(stream,Loader=yaml.Loader)
    
    min_len = min(len(data["point_data"]),len(data["vocals"]))
    scaler = StandardScaler()

    take_len = 30*10
    X_data = {}
    y_data = []

    for csn in class_name :
        X_data[csn] = []
        scaler = scaler.fit(np.array(data[csn]).reshape(-1, 1))
        taget_data = scaler.transform(np.array(data[csn]).reshape(-1, 1))
        for i in range( take_len , min_len ) :
            X_data[csn].append( [ i[0] for i in taget_data[ i-take_len : i]] )

    for i in range( take_len , min_len ) :
        y_data.append( list(np.array(data["point_data"][ i-take_len : i])*100000) )

    for i in range(len(y_data)) :
        tmp = []
        for j in range( 300 ) :
            tmp.append( [ X_data[csn][i][j] for csn in class_name] )
        X_save.append(tmp)

    return X_save,y_data


def main() :

    X_data = []
    y_data = []

    for file_name in  tqdm(os.listdir(data_path)) :
        X_t,y_t =  get_data(file_name)
        X_data += X_t
        y_data += y_t

    print("total data num : {}".format(len(y_data)))

    train_X = K.cast_to_floatx(np.array( X_data ))
    train_Y = K.cast_to_floatx(np.array( y_data ))
    
    X_train, X_test, y_train, y_test = train_test_split( train_X , train_Y , test_size=0.33 , random_state=42 )

    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_freq=10, save_weights_only=True)
    model = buildManyToManyModel()
    model.fit(X_train, y_train , epochs=500, batch_size=128,
              validation_data=(X_test,y_test),validation_batch_size=128,
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
    #X_save,y_data = get_data("[乖離性ミリオンアーサー][Million Ways=One Destination][NEET].yml")
    #print(X_save[0])
    #rint(y_data[0])