import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import to_categorical  #원핫 인코딩 하기 위해 쓰임
from keras.models import Sequential     #model은 functional API와 Sequential 두가지를 제공해주지만 다층 퍼셉트론(왼->오)에서는 sequential 사용
from keras.layers import Dense          #다층 퍼셉트론을 구성하는 완전 연결층은 Dense 클래스로 쌓음
from keras.optimizers import Adam       #SGD를 학습 알고리즘으로 사용

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)
x_train = x_train.astype(np.float32)/255.0          #0~1로 정규화
x_test = x_test.astype(np.float32)/255.0            #0~1로 정규화
y_train = to_categorical(y_train,10)                #원핫코딩
y_test = to_categorical(y_test,10)  

dmlp = Sequential()
dmlp.add(Dense(units = 1024,activation = 'relu',input_shape=(3072,)))
dmlp.add(Dense(units = 512,activation='relu'))
dmlp.add(Dense(units = 512,activation='relu'))
dmlp.add(Dense(units = 10, activation='softmax'))

dmlp.compile(loss='categorical_crossentropy',optimizer = Adam(learning_rate=0.0001),metrics=['accuracy'])
hist = dmlp.fit(x_train,y_train,batch_size = 128,epochs=50,validation_data=(x_test,y_test),verbose=2)
print('정확률=',dmlp.evaluate(x_test,y_test,verbose=0)[1]*100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.grid()
plt.show()