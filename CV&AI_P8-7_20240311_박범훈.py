from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Rescaling
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121     #DenseNet121를 모델로 사용
from keras.utils import image_dataset_from_directory    #폴더에서 영상을 읽기위해서 사용
import pathlib  #폴더 사용하기 위한 모듈

data_path = pathlib.Path(r'C:\Users\IT\Desktop\CV&AI\img\datasets\Stanford_dogs\images\images') #Standofrd Dog의 데이터 셋이 있는 폴더 지정

train_ds=image_dataset_from_directory(data_path,validation_split=0.2, subset = 'training',seed=123, image_size=(224,224),batch_size = 16)   #훈련 데이터 불러옴
test_ds =image_dataset_from_directory(data_path,validation_split=0.2, subset = 'validation',seed=123, image_size=(224,224),batch_size = 16) #테스트 데이터 불러옴

base_model=DenseNet121(weights = 'imagenet',include_top=False, input_shape=(224,224,3)) 
cnn=Sequential()
cnn.add(Rescaling(1.0/255.0))
cnn.add(base_model)                             #base모델설정
cnn.add(Flatten())                          
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.75))                          #base 모델 뒤에 추가적으로 미세 조정을 위한 신경망 구성
cnn.add(Dense(units=120,activation='softmax'))

cnn.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.000001),metrics=['accuracy'])
hist=cnn.fit(train_ds,epochs=200,validation_data=test_ds,verbose=2)

print('정확률=',cnn.evaluate(test_ds,verbose=0)[1]*100)

cnn.save('cnn_for_standford_dogs')

import pickle
f=open('dog_species_names.txt','wb')
pickle.dump(train_ds.class_names,f)
f.close()

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()