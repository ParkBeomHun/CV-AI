import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Conv2D, MaxPooling2D, Flatten, Dropout

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 동적 할당 활성화
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


if tf.test.is_gpu_available():
    print('GPU를 사용하여 TensorFlow가 학습 중입니다.')
else:
    print('CPU를 사용하여 TensorFlow가 학습 중입니다.')

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32)/255.0          #0~1로 정규화
x_test = x_test.astype(np.float32)/255.0            #0~1로 정규화
y_train = to_categorical(y_train,10)                #원핫코딩
y_test = to_categorical(y_test,10)                  #원핫코딩

strategy = tf.distribute.MirroredStrategy()
with tf.device('/GPU:0'):
    cnn = Sequential()
    cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
    cnn.add(Conv2D(32,(3,3),activation='relu'))
    cnn.add(MaxPooling2D(pool_size = (2,2),strides=2))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(64,(3,3),activation='relu'))
    cnn.add(Conv2D(64,(3,3),activation='relu'))
    cnn.add(MaxPooling2D(pool_size = (2,2),strides=2))
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())
    cnn.add(Dense(units=512,activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(units=10,activation='softmax'))


    cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
    hist = cnn.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=(x_test,y_test),verbose=2)

    res=cnn.evaluate(x_test,y_test,verbose=0)
    print('정확률 = ',res[1]*100)

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