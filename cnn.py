# -*- coding: utf-8 -*-

###############################################################################
#
#   BUILD CNN
#
###############################################################################

"""
The architecture:
    - Input layer: 32 x 32 RGB images
    - Convolutional layer: 32 filters, 3 X 3 kernel, ReLU activation
    - Pooling layer: Max Pooling, 2 x 2 filter size
    - Dropout layer: rate = 0.25
    - Flatten layer & input to ANN
    - Fully connected layer: 64 nodes, ReLU activation
    - Dropout layer: rate = 0.5
    - Output layer: sigmoid activation
    
    - OPTIMIZER: adam
    - LOSS: binary cross entropy
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

# Convolutional layer
classifier.add(Convolution2D(filters = 32, kernel_size = (3, 3), input_shape = (32, 32, 3), activation = 'relu')) 
 
# Pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Dropout
classifier.add(Dropout(rate = 0.25))

# Flattening layer
classifier.add(Flatten())

# Fully connected
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate = 0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


###############################################################################
#
#   AUGMENT AND PROCESS DATA
#
###############################################################################

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create data for training
train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=3,
        validation_data=test_set,
        validation_steps=2000)


###############################################################################
#
#   SINGLE PREDICTION
#
###############################################################################

import numpy as np
from keras.preprocessing import image

# Load test image
test_image = image.load_img('dataset/single_prediction/maybe_cat.jpg', target_size=(32, 32))
test_image = image.img_to_array(test_image)

# Add extra dimension to the image array to correspond to the batch -- tedious
# but necessary
test_image = np.expand_dims(test_image, axis = 0)

# Predict on test image
result = classifier.predict(test_image)

# Map result to the class index it refers to
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'




