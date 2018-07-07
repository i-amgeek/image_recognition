#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 23:40:12 2018

@author: rishabhshridhar
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


classifier=Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(256,256,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(64,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255,
               shear_range=0.2,
               zoom_range=0.2,
               horizontal_flip=True
               )
test_datagen=ImageDataGenerator(rescale=1./255)

trainingset=train_datagen.flow_from_directory(r'C:\Users\Z003Y61B\Desktop\Convolutional_Neural_Networks\dataset\training_set',
                                              target_size=(256,256),
                                              batch_size=32,
                                              class_mode='binary'
                                              )
               
testset=test_datagen.flow_from_directory(r'C:\Users\Z003Y61B\Desktop\Convolutional_Neural_Networks\dataset\test_set',
                                              target_size=(256,256),
                                              batch_size=32,
                                              class_mode='binary'
                                              )
classifier.fit_generator(trainingset,steps_per_epoch=(8000/32), epochs=80,validation_data=testset,
                         validation_steps=(2000/32)
                         )

classifier.save('/Users/rishabhshridhar/Desktop/Projects/Image recognition/FCNN.h5')

classifier=load_model("/Users/rishabhshridhar/Desktop/Projects/Image recognition/FCNN.h5")


pic=image.load_img(r'/Users/rishabhshridhar/Desktop/Projects/Image recognition/n.jpg',target_size=(256,256))
pic=image.img_to_array(pic)
pic=np.expand_dims(pic,axis=0)
s=classifier.predict(pic)
if(s):
    print("dog")
else:
    print("cat")
