import numpy as np
import tensorflow as tf
from keras.datasets import mnist

from keras.utils import to_categorical  #원핫 인코딩 하기 위해 쓰임
from keras.models import Sequential     #model은 functional API와 Sequential 두가지를 제공해주지만 다층 퍼셉트론(왼->오)에서는 sequential 사용
from keras.layers import Dense          #다층 퍼셉트론을 구성하는 완전 연결층은 Dense 클래스로 쌓음
from keras.optimizers import Adam        #SGD를 학습 알고리즘으로 사용

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
x_train = x_train.astype(np.float32)/255.0          #0~1로 정규화
x_test = x_test.astype(np.float32)/255.0            #0~1로 정규화
y_train = to_categorical(y_train,10)                #원핫코딩
y_test = to_categorical(y_test,10)                  #원핫코딩

mlp = Sequential()  #sequential 함수로 mlp 객체 생성
mlp.add(Dense(units = 512, activation = 'tanh', input_shape = (784,)))  #입력노드 784개 / 활성함수 tanh / 은닉층 노드수 512 / FC(DENSE)    : 은닉층 추가
mlp.add(Dense(units = 10,activation = 'softmax'))                       #출력층 추가

mlp.compile(loss='MSE', optimizer = Adam(learning_rate=0.001),metrics=['accuracy'])               #loss 함수로 평균오차제곱 / 학습률 0.01 / SGD 학습 알고리즘 사용 / 정확률을 기준으로 성능 측정
mlp.fit(x_train,y_train,batch_size = 128,epochs=50,validation_data=(x_test,y_test),verbose=2)   #

res = mlp.evaluate(x_test,y_test,verbose=0)
print("정확률 : ",res[1]*100)