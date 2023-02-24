from sklearn import datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dense,Dropout
from keras.utils import to_categorical 

def main() :
    iris = datasets.load_iris()
    
    iris_X = iris.data
    iris_Y = iris.target
    print(len(iris_X))
    print(len(iris_Y))
    

    X_train, X_test, y_train, y_test = train_test_split( iris_X, iris_Y , train_size=0.8 )
    #X_train = X_train.reshape()

    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=100)
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.Dense(1, use_bias=False,input_shape=(2, )))
    #model.summary()

    model = tf.keras.models.Sequential([
        Dense(units=8,input_dim=4,use_bias=True,activation='sigmoid'),
        Dense(units=16,use_bias=True,activation='sigmoid'),
        Dropout(0.2),
        Dense(units=3,activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    model.fit(X_train, y_train,batch_size=32 ,epochs=1000,callbacks = [callback])
    
    model.evaluate(X_test, y_test)
    
    y_pred = model.predict(X_test[:1])

    print(y_test[0])
    print(np.argmax(y_pred, axis=1))
if __name__ == "__main__" :
    main()