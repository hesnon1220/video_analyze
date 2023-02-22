import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = []
with open( r"F:\work\video_analyze\output\score.txt" , "r") as txtfile :
    lines = txtfile.readlines()
    for i in lines :
        data.append( list(map(float,i.replace("\n","").split("\t"))) )


print(data)

train_data, test_data = train_test_split(data, train_size=0.9)



train_X = []
train_y = []
for i in train_data :
    train_X.append(i[:3])
    train_y.append(i[-1])

test_X = []
test_y = []
for i in test_data :
    test_X.append(i[:3])
    test_y.append(i[-1])

linearModel = LinearRegression()
linearModel.fit(train_X, train_y)

y_pred = linearModel.predict(test_X)

print('MSE:', mean_squared_error(y_pred, test_y))
"""
# 訓練模型
linearModel = LinearRegression()
linearModel.fit(X, y)

y_pred = linearModel.predict(X)
# 21.894831181729202
print('MSE:', mean_squared_error(y_pred, y))
"""