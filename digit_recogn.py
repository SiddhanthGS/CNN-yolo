import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
IMG_SIZE=28
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#plt.imshow(x_train[0])
#plt.show()
#print(x_train[0])
x_train=tf.keras.utils.normalize(x_train,axis=1)#normalise the values,intiially from 0-255 now 0-1
x_test=tf.keras.utils.normalize(x_test,axis=1)
#print(y_train[0])
x_trainr=np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)#-1 corresponds to 60000,i.e. max 
x_testr=np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)#1 is for adding another extra dimension to our images,also -1 corresp to 10000 here(max again)
model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:]))#64 diff kerneles used,also dim start from 1 bcos index 0 is for image which is not needed
model.add(Activation("relu"))#to make img non linear
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())#2d to 1d conversion
model.add(Dense(64))#64 neurons ka layer is added
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))#bcos output me 0-9 ke hi nodes honge
model.add(Activation("softmax"))#sigmoid can also be used
#print(model.summary)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(x_trainr,y_train,epochs=5,validation_split=0.3)
test_loss,test_acc=model.evaluate(x_testr,y_test)
predictions=model.predict([x_testr])
for i in range(10):
    print(np.argmax(predictions[i]))
    plt.imshow(x_test[i])
    plt.show()