import cv2
import numpy as np
import functions as fcns
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

X_train, y_train = fcns.loadallimgs()

print(X_train.shape)
print(y_train.shape)

fcns.histplot(y_train, 'figure1.jpg')

X_train_aug, y_train_aug = fcns.data_augment(X_train, y_train)

X_train = np.concatenate((X_train, X_train_aug), axis=0)
y_train = np.concatenate((y_train, y_train_aug), axis=0)

fcns.histplot(y_train, 'figure2.jpg')


'''
#Model architecture

# Simple trial architecture
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
# model.add(Flatten())
# model.add(Dense(1))



# #Some constants
# image_size = [32,32]
# crop_size = [[70,25], [0,0]]

# #Sample image preprocessing
# #Determine how many pixels to crop from images
# img = X_train[1]
# print(img.shape)
# fcns.plot_img(img,str = 'Before crop')
# img = fcns.crop(img, crop_size)
# print(img.shape)
# fcns.plot_img(img, str = 'After crop')

#Nvidia end-to-end deep learning CNN architecture	
model = Sequential()
#model.add(Lambda(lambda x: fcns.crop(x, crop_size), input_shape= (160,320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
#model.add(Lambda(lambda x: fcns.resize(x, image_size))
model.add(Convolution2D(24, 5, 5, subsample= (2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample= (2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample= (2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch = 5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')

'''