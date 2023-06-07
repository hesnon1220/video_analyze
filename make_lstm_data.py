import yaml
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import threading


output_path = r"F:\work\video_analyze\my_work\output"
data_path = r"F:\work\video_analyze\my_work\train_data"
class_name = ['bass', 'drums', 'other', 'vocals']


def task(file_yml) :

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
            X_data[csn].append( taget_data[ i-take_len : i].reshape(1, -1) )

    for i in range( take_len , min_len ) :
        y_data.append( data["point_data"][ i-take_len : i] )

    
    for i in range(len(y_data)) :
        tmp = []
        for csn in class_name :
            tmp.append( X_data[csn][i] )
        X_save.append(tmp)


    with open(os.path.join( output_path , "X" , "{}.yml".format(file_name)), 'w') as f:
        yaml.dump(X_save, f)
    with open(os.path.join( output_path , "y" , "{}.yml".format(file_name)), 'w') as f:
        yaml.dump(y_data, f)

class MyThread(threading.Thread):
    def __init__(self, file_yml,semaphore):
        threading.Thread.__init__(self)
        self.file_yml = file_yml
        self.semaphore = semaphore
    ####################################################################################################################################
    def run(self):
        with self.semaphore :
            task(self.file_yml)

def main():

    max_deals = 16
    semaphore = threading.BoundedSemaphore(max_deals)
    threads = []
    for file_yml in os.listdir(data_path) :
        threads.append(MyThread(file_yml,semaphore))
    ####################################################################################################################################
    for _idx_ in range(len(threads)) :
        threads[_idx_].start()
    ####################################################################################################################################
    for _idx_ in range(len(threads)):
        threads[_idx_].join()



if __name__ == "__main__" :
    main()