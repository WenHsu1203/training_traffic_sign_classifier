# Load pickled data
import pickle

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

import numpy as np
# Number of training examples
n_train = len(X_train)
# Number of validation examples
n_validation = len(X_valid)
# Number of testing examples.
n_test = len(X_test)
# What's the shape of an traffic sign image?
image_shape = X_train[0].shape
# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes= 43)
y_valid = to_categorical(y_valid, num_classes= 43)
y_test = to_categorical(y_test, num_classes= 43)

import cv2
def img_to_tensor(img):
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	# resize the image to (32, 32, 1)
    img = cv2.resize(img,(32,32))
    # convert 3D tensor to 4D tensor with shape (1, 32, 32, 1) and return 4D tensor
    return np.expand_dims(img, axis=0)

def imgs_to_tensor(imgs):
    list_of_tensors = [img_to_tensor(img) for img in imgs]
    return np.vstack(list_of_tensors)
# to tensors and normalize it
train_tensors = imgs_to_tensor(X_train).astype('float32')/255
valid_tensors = imgs_to_tensor(X_valid).astype('float32')/255
test_tensors = imgs_to_tensor(X_test).astype('float32')/255
# print('----------------------------')
# print('shape:', np.expand_dims(train_tensors[0], axis=0).shape)

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint  
from keras import applications
from keras import optimizers
from keras.models import load_model

model = Sequential()
# 32,32,16
model.add(Conv2D(filters = 16, kernel_size = 5, padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
# 16,16,16
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# 16,16,32
model.add(Conv2D(filters = 32, kernel_size = 5, padding = 'same', activation = 'relu'))
# 8,8,32
model.add(MaxPooling2D(pool_size = 2))
# Dropout
model.add(Dropout(0.5))
# flatten
model.add(Flatten())
# 512
model.add(Dense(512,activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
# 128
model.add(Dense(128, activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
# Fully connected Layer to the number of signal categories
model.add(Dense(43, activation = 'softmax'))

model.summary()

# compile the model
model.compile(loss = "categorical_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

# train the model.
epochs = 10
batch_size = 32

checkpointer = ModelCheckpoint(filepath='weights.h5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, y_train, 
          validation_data=(valid_tensors, y_valid),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

del model
model = load_model('weights.h5')
signal_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# print out test accuracy
test_accuracy = 100*np.sum(np.array(signal_predictions)==np.argmax(y_test, axis=1))/len(signal_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
