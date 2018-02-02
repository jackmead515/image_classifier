# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator



##################################################################################################################
##################################################################################################################
##################################################################################################################
'''
Step 1 --- Initialising the CNN

This is the basic model for a CNN to which you can add activation layers too.
'''
classifier = Sequential()



##################################################################################################################
##################################################################################################################
##################################################################################################################
'''
Step 2 --- Adding the Convolution --- https://keras.io/layers/convolutional/

This is the first layer of the network: the 2d convolution layer. (for 2d images)
The conv layer will take the ORIGINAL input images and apply an activation function
acrossed it. The activation function will detector the features to give to the next layer.

filters -> the number output of filters in the convolution
kernel_size -> a tuple or integer expressing the dimensions of the feature detector
input_shape ->
    0 index --> x/width dimension of input image
    1 index --> y/height dimension of input images
    2 index --> integer describing layers of colors. 3 = rgb, 2 = black/white
activation -> activation function to apply. In this case, the 'relu' function.
'''
classifier.add(Conv2D( filters = 32 , kernel_size = 3 , input_shape = (64, 64, 3) , activation = 'relu' ))



##################################################################################################################
##################################################################################################################
##################################################################################################################
'''
Step 3 --- Pooling --- https://keras.io/layers/pooling/

This layer is used to reduce the image size while selecting the highest
values from it's feature detector. It's main goal is to downsample the data
to further prep it to be pushed through a traditional neural network

pool_size ->
    0 index --> x/width of feature detector
    1 index --> y/height of feature detector
'''
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D( filters = 32 , kernel_size = 3 , activation = 'relu' ))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



##################################################################################################################
##################################################################################################################
##################################################################################################################
'''
Step 4 --- Flattening

This layer is really straight forward. It reduces a vector of input values to a
single dimensional vector. Essentially, it takes a 2D array and flattens it to a 1D array.
This is needed to push the values through the first layer of the neural net.
'''
classifier.add(Flatten())



##################################################################################################################
##################################################################################################################
##################################################################################################################
'''
Step 5 --- First Layer of CNN --- https://keras.io/layers/core/

Adds the official first layer of the neural network to the AI.

units -> dimension of output numbers. This means there will be 128 values coming out of this layer.
activation -> The activation function to apply to this layer.
'''
classifier.add(Dense(units = 128, activation = 'relu'))



##################################################################################################################
##################################################################################################################
##################################################################################################################
'''
Step 5 --- Output Layer of CNN --- https://keras.io/layers/core/

Adds the official ouput layer of the neural network to the AI. This output only
needs 1 output node as this will essentially be our "yes" or "no".

units -> dimension of output numbers. This means there will be 128 values coming out of this layer.
activation -> The activation function to apply to this layer.
'''
classifier.add(Dense(units = 1, activation = 'sigmoid'))



##################################################################################################################
##################################################################################################################
##################################################################################################################
'''
Step 6 --- Compiling the CNN --- https://keras.io/models/sequential/

This step hooks all layers together and adds in an optimizer and loss function to
the CNN to learn from.

optimizer -> optimizer function to use on the output. https://keras.io/optimizers/
loss -> the loss function to use on the output. https://keras.io/losses/

'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



##################################################################################################################
##################################################################################################################
##################################################################################################################
'''
Step 7 --- Fitting Training Data to CNN --- https://keras.io/preprocessing/image/

This step is all about gathering the training and testing images to push through the CNN.
The images have to be shaped and fitted so the CNN will accept them properly.
'''

# Creating the Training Set Generator
# rescale -> Value to multiple RGB (values from 1-255) by. 1.0/255 is a typical value to use
# shear_range -> Randomly applys shearing transformations. Float represents intensity of shearing.
# zoom_range -> Randomly zooming into images.
# horizontal_flip -> Randomly flips HALF of the images horizontally.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# Creating the Test Set Generator
test_datagen = ImageDataGenerator(rescale = 1./255)

# Loading the (partial in batch_sizes) Training Set into memory.
# directory -> Directory in which the images reside.
# target_size -> Dimensions of which each image will be resized too.
# batch_size -> The amount of images to train in one push-through of the CNN. Not to be confused with an epoch
# class_mode ->  Determines the type of label arrays that are returned
training_set = train_datagen.flow_from_directory(directory = 'dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Loading the (partial in batch_sizes) Test Set into memory
test_set = test_datagen.flow_from_directory(directory = 'dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Fits the training_set and test_set to the CNN.
# steps_per_epoch -> typically equal to the number of samples of your dataset divided by the batch_size
# epochs -> How man epochs to run the traning_set on
# validation_data -> The test set
# validation_steps -> Should be equal to the number of samples of test set divied by the batch_size
classifier.fit_generator(training_set,
                         steps_per_epoch = 300,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 100)

# Saves to the weights of a model to the file specified
classifier.save_weights('model_1.h5');
