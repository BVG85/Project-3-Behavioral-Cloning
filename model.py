############# Model file for Behavioral Cloning Project

import csv
import cv2
import numpy as np
import tensorflow as tf

############# Import of data recorded in simulator

lines = []

with open('./data/driving_log.csv') as csvfile: 
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


images = []
measurements = []

correction_factor = 0.3 ## Correction factor for left and right camera images

for line in lines:
	for i in range(3):
		
		########## Import of center camera images

		if (i==0): 
			source_path = line[0]
			filename = source_path.split('/')[-1]
			#print(filename)
			current_path = './data/IMG/' + filename
			image = cv2.imread(current_path)
			images.append(image)
			steering_angle = float(line[3])
			measurements.append(steering_angle)

		########## Import of left camera images

		elif (i==1):
			source_path = line[1]
			filename = source_path.split('/')[-1]
			#print(filename)
			current_path = './data/IMG/' + filename
			image = cv2.imread(current_path)
			images.append(image)
			steering_angle = float(line[3]) + correction_factor
			measurements.append(steering_angle)

		########## Import of right camera images

		elif (i==2):
			source_path = line[2]
			filename = source_path.split('/')[-1]
			#print(filename)
			current_path = './data/IMG/' + filename
			image = cv2.imread(current_path)
			images.append(image)
			steering_angle = float(line[3]) - correction_factor
			measurements.append(steering_angle)


X_train = np.array(images)
y_train = np.array(measurements)


########################################################

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers import Cropping2D

from keras.layers.pooling import MaxPooling2D
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

########################################################

model = Sequential()

### Normalization ###

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Flatten(input_shape=(160,320,3)))

### NVIDEA architecture ###

model.add(Cropping2D(cropping=((75,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(0.8))
model.add(Dense(50))
#model.add(Dropout(0.8))
model.add(Dense(10))

###

model.add(Dense(1))

### Used TensorBoard for model visualization

tensorboard = TensorBoard(log_dir='./logt', histogram_freq=0,write_graph=True,write_images=False)

model.compile(loss='mse', optimizer='adam')


history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, verbose=1,callbacks=[tensorboard])


#print(history_object.history.keys())

#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()


model.summary()
model.save('model.h5')