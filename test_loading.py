import pickle
import numpy as np

testing_file = 'test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    

X_test, y_test = test['features'], test['labels']

from keras.utils import to_categorical

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
test_tensors = imgs_to_tensor(X_test).astype('float32')/255

from keras.models import model_from_json

# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('weights_2.h5')

signal_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# print out test accuracy
test_accuracy = 100*np.sum(np.array(signal_predictions)==np.argmax(y_test, axis=1))/len(signal_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)