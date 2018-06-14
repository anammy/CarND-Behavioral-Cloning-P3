import functions as fcns
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras import optimizers
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import csv
import random

#Path to training dataset provided
log_path = './data/data/driving_log.csv'
img_path = './data/data/IMG/'

#Path to training data generated by me
#log_path = './training_data/driving_log.csv'
#img_path = ''

#Read the image paths and steering angle data from the csv file
samples = []
with open(log_path) as csvfile:
	reader = csv.reader(csvfile)
	first_line = next(reader)
	for line in reader:
		samples.append(line)

#Split the training dataset into training and validation data sets. The model generated will be then be tested in the simulator using drive.py
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Generator function to load training and validation images in batches
def generator(samples, batch_size=32):
	num_samples = len(samples)
	path = img_path
	'''
	#Parameters for plots of data augmentation
	iter=0
	maxgraph=20
	condition = True
	'''
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				#Load images from all 3 cameras
				center_image = cv2.imread(path + batch_sample[0].split('/')[-1])
				left_image = cv2.imread(path + batch_sample[1].split('/')[-1])
				right_image = cv2.imread(path + batch_sample[2].split('/')[-1])
				#Convert images from BGR to RGB since the latter is used in drive.py
				center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
				right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
				left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
				center_angle = float(batch_sample[3])
				correction = 0.2 # this is a parameter to tune
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				images.append(center_image)
				angles.append(center_angle)
				images.append(left_image)
				angles.append(left_angle)
				images.append(right_image)
				angles.append(right_angle)
				
				#Data augmentation by flipping and translating images for steering angles not well represented in the training dataset
				if (center_angle > 0.3) or (center_angle < -0.3) or (center_angle > 0 and center_angle < 0.2) or (center_angle > -0.2 and center_angle < 0):
					images.append(np.fliplr(center_image))
					angles.append(-1.0*center_angle)
					for i in range(3):
						pixel_x = random.randint(-3, 4)
						pixel_y = random.randint(-3, 4)
						img_translate = fcns.translate(center_image, pixel_x, pixel_y)
						images.append(img_translate)
						angles.append(center_angle)
			
			X_train = np.array(images)
			y_train = np.array(angles)
			
			'''
			#For plots of data augmentation
			iter+=1
			if iter == maxgraph:
				condition=False
			if condition:
				fcns.histplot(y_train, 'gen'+str(iter)+'.jpg')
			'''
			
			yield shuffle(X_train, y_train)

			

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#Nvidia end-to-end deep learning CNN architecture with dropout layers	
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample= (2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample= (2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample= (2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

#Set learning rate of adam optimizer
ADAM =  optimizers.adam(lr=0.001)

model.compile(loss='mse', optimizer=ADAM)

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=10, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./figures/mse.jpg')
plt.show()

model.save('model.h5')