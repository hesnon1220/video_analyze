import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
import cv2
from tqdm import tqdm
from moviepy.editor import AudioFileClip,VideoFileClip
from Helper_private import *
import wave
import pylab as pl
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import cluster
import re
import random

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

class Model(nn.Module):
    def __init__(self,array_len):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(array_len*919, 2*array_len*919,bias=True)  # 4x25 input features, 8 output features
        self.linear2 = nn.Linear(2*array_len*919, 2*array_len,bias=True)
        self.linear3 = nn.Linear(2*array_len, array_len,bias=True)
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten the input tensor along the second dimension
        x = torch.sigmoid(self.linear1(x))  # Apply ReLU activation to output of first layer
        x = torch.sigmoid(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x

def make_tensor(X_train,y_train,X_val,y_val,device) :
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    return X_train,y_train,X_val,y_val



def train(X_train, y_train, X_val, y_val,array_len):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_val, y_val = make_tensor(X_train, y_train, X_val, y_val,device)

    model = Model(array_len).to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    # Train the model
    num_epochs = 100
    batch_size = 32


    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

        # Evaluate the model on the validation set
        with torch.no_grad():
            outputs = model(X_val)
            #print(outputs)
            #_, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total = y_val.size(0)
            #print(predicted.shape)
            #print(y_val.shape)
            correct = (outputs == y_val).sum().item()
            accuracy = 100 * correct / total / y_val.size(1)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")


def main() :

    fps = 23.98

    #audio_path = os.path.join(r"F:\work\video_analyze\separated\htdemucs\test",i)
    audio_path = r"F:\work\video_analyze\data\audio\Beelzebub-jou no Okinimesu Mama\separated\htdemucs\01.ピンクレモネード\vocals.wav"
    y,sr = librosa.load(audio_path)
    totla_frame = int(len(y)/sr*fps)
    fft_data = get_fft(audio_path,fps)
    with open(r"F:\work\video_analyze\output\lnc_time.txt","r") as txt_file :
        line = txt_file.readline()
        lnc_time = np.array(list(map(float,line.replace("\r","").split("\t"))))
    lnc_time = np.array(lnc_time*fps,dtype = "uint64")
    target = np.zeros([totla_frame])
    

    for i in lnc_time :
        target[i] = 1


    array_len = 10

    X_data = []
    Y_data = []
    for i in range(len(target)-array_len) :
        if max(target[i:i+array_len]) != 1 : continue
        tmp_array = np.array(fft_data[i:i+array_len][:])
        X_data.append( tmp_array  )
        tmp_target = target[i:i+array_len]
        k = np.where(tmp_target == 1)[0][0]
        tmp_target[k:] = 1
        Y_data.append( tmp_target )

    X_train, X_val, y_train, y_val = train_test_split( np.array(X_data), np.array(Y_data) , train_size=0.8 )

    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)


    train(X_train, y_train, X_val, y_val,array_len)

if __name__ == "__main__" :
    main()