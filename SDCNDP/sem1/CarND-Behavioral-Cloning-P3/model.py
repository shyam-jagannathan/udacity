import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.utils.visualize_util import plot
from keras.optimizers import SGD, Adam

########################################################################
# Input Helper functions defined here
########################################################################
def readTrainSet(data):
    train_set = []
    for i in range(len(data)):
        img = cv2.imread(data[i])
        mono = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mono = cv2.resize(mono, (64, 80), interpolation=cv2.INTER_AREA)
        train_set.append(mono)
    
    return np.array(train_set)

def flipImage(data):
    #print('Before flip...')
    #plt.imshow(data[0])
    #plt.show()
    
    data_flip = []
    for i in range(len(data)):
        data_flip.append(np.fliplr(data[i]))
    
    #print('After flip...')
    #plt.imshow(data_flip[0])
    #plt.show()
    
    return np.array(data_flip)

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def getData(run_list, file_name):
    
    print('Reading list =>', run_list[0] + file_name)
        
    tbl = pd.read_csv(run_list[0] + file_name)
    tbl_arr = np.array(tbl)
    
    center_img = tbl_arr[:, 0]
    steering_angle = tbl_arr[:, 3]
    
    for i in range(1, len(run_list)):
        print('Reading list =>', run_list[i] + file_name)
        
        tbl = pd.read_csv(run_list[i] + file_name)
        tbl_arr = np.array(tbl)
        tmp_img = tbl_arr[:, 0]
        tmp_ang = tbl_arr[:, 3]
        center_img = np.concatenate([center_img, tmp_img])
        steering_angle = np.concatenate([steering_angle, tmp_ang])
    
    return center_img, steering_angle

#######################################################################
# Image Preprocessing done here
#######################################################################
def prepareInputs(run_list):
	X_train, y_train = getData(run_list, 'driving_log.csv')

	print('Reading TrainSet...')
	X_train = readTrainSet(X_train)
	print('Done. Read - ', X_train.shape)

	print('Flip data set..')
	X_train_flip = flipImage(X_train)
	y_train_flip = -y_train
	print('Done')

	X_train = np.concatenate((X_train, X_train_flip))
	y_train = np.concatenate((y_train, y_train_flip))

	print('Crop Data to 64x64...')
	X_train = X_train[:, :64, :]
	print(X_train.shape)
	print('Done')

	print('Shuffle dataSet...')
	X_train, y_train = shuffle(X_train, y_train) 
	print('Done')

	print('Normalize Data...')
	X_normalized = normalize_grayscale(X_train)
	print('Done')

	return X_normalized, y_train

########################################################################
# Network model built here
########################################################################
def networkModel():

	model = Sequential()
	#Layer 1 - 5x5 conv, Input=(64x64x3), Output(60x60x16)
	model.add(Convolution2D(16, 5, 5, input_shape=(64, 64, 3)))
	#Maxpooling 2x2, stride 2, Input=(60x60x16), Output=(30x30x16)
	model.add(MaxPooling2D((2, 2)))
	#Activation - ReLU
	model.add(Activation('relu'))

	#Layer 2 - 3x3 conv, Input=(30x30x16), Output(28x28x32)
	model.add(Convolution2D(32, 3, 3, input_shape=(30, 30, 16)))
	#Maxpooling 2x2, stride 2, Input=(28x28x32), Output=(14x14x32)
	model.add(MaxPooling2D((2, 2)))
	#Activation - ReLU
	model.add(Activation('relu'))

	#Layer 3 - 3x3 conv, Input=(14x14x32), Output(12x12x64)
	model.add(Convolution2D(64, 3, 3, input_shape=(14, 14, 32)))
	#Maxpooling 2x2, stride 2, Input=(12x12x64), Output=(6x6x64)
	model.add(MaxPooling2D((2, 2)))
	#Activation - ReLU
	model.add(Activation('relu'))

	#Layer 4 - 3x3 conv, Input=(6x6x64), Output(4x4x128)
	model.add(Convolution2D(128, 3, 3, input_shape=(6, 6, 64)))
	#Maxpooling 2x2, stride 2, Input=(4x4x128), Output=(2x2x256)
	model.add(MaxPooling2D((2, 2)))
	#Activation - ReLU
	model.add(Activation('relu'))

	#Layer 5- Input=(2x2x256), Output=1024
	model.add(Flatten())

	#Layer 6 - Input=1024, Output=512
	model.add(Dense(512))
	#Activation - Tanh
	model.add(Activation('tanh'))
	
	#Layer 7 - Input=512, Output=128
	model.add(Dense(128))
	#Activation - Tanh
	model.add(Activation('tanh'))
	#Dropout - 50%
	model.add(Dropout(0.5))
	
	#Layer 8 - Input=128, Output=32
	model.add(Dense(32))
	#Activation - Tanh
	model.add(Activation('tanh'))
	#Dropout - 50%
	model.add(Dropout(0.5))
	
	#Layer 9 - Input=32, Output=1
	model.add(Dense(1))
	model.add(Activation('tanh'))
	
	#Display network architecture
	plot(model, to_file='model.png')

	return model


if __name__ == '__main__':

	# List of training samples
	run_list = ["/home/shyam/Work/SDCNDP/dataset/Track1/run1/",
	            "/home/shyam/Work/SDCNDP/dataset/Track1/run2/",
	            "/home/shyam/Work/SDCNDP/dataset/Track1/run3/",
	            "/home/shyam/Work/SDCNDP/dataset/Track1/run4/"]

	# Preprocess inputs
	X_train, y_train = prepareInputs(run_list)

	# Setup Network
	model = networkModel()

	# Train the network
	#Use Adam optimizer and loss function as mean-square-error. 
	#Default init learning rate is 0.001
	model.compile('adam', 'mse')

	#Supply the normalized input and steering angles, batch size to 256, num epochs = 100, 20% train-validation split
	history = model.fit(X_train, y_train, batch_size=256, nb_epoch=100, validation_split=0.2)

        # Save the model
	model.save('model.hd5')

        print('Training complete!')










