# CNN-Example
This is an example script to create, train, and test a convolutional neural network.

# Problem Statement
Distinguish between pictures of cats and dogs.

# Performance
The trained CNN scores a baseline accuracy of 75% on the evaluation dataset. Due to computation and time constraints, no further analysis of
its scoring was possible. Ideally, we would perform k-fold cross validation with no fewer than 10 folds. From the
performances collected, we would calculate a mean accuracy and the variance of the scores. 

# Data Set
The data set was 10,000 images of cats and dogs, divided into 5,000 cats and 5,000 dogs. The dataset was partitioned into 
4,000 training samples from each class and 1,000 testing samples from each class.

The dataset comes presegmented into a testing set and a training set. Using the `ImageDataGenerator` class from Keras, we were
able to efficiently load the data into the environment. The directory structure for the dataset is as follows:
```
/project_root
    |
    |
    |_ dataset/
        |
        |_ single_prediction/
        |
        |
        |_ test_set/
        .   |
        .   |_ cats/
        .   |
        .   |_ dogs/
        .
        |
        |_ training_set/
            |
            |_ cats/
            |
            |_ dogs/ 
```
# General Architecture
- Input layer: 32 x 32 RGB images
- Convolutional layer: 32 filters, 3 X 3 kernel, ReLU activation
- Pooling layer: Max Pooling, 2 x 2 filter size
- Dropout layer: rate = 0.25
- Flatten layer & input to ANN
- Fully connected layer: 64 nodes, ReLU activation
- Dropout layer: rate = 0.5
- Output layer: sigmoid activation

- OPTIMIZER: adam
- LOSS FN: binary cross entropy
- BATCH SIZE: 32
- EPOCHS: 3

# Why the aggressive dropout?
During the first few iterations of the net's architecture, we noticed that the net was performing quite well on the training
set with accuracies around 96%. Its performance on the testing set was around 70%, however. We immediately recognized this
as a symptom of overfitting. That is, the net was beginning to learn correlations in the training set and failing to generalize
this learning to a different set of data. To compensate for this overfitting, we applied some aggressive dropout layers between
the convolutional layers and fully connected layer. This improved consistency between training set and testing set performances, 
with final performance being around 80% on the training set and 75% on the testing set. In some epochs, the accuracy on the
testing set was even higher than the accuracy on the training set.  

# Environment
Python 3.6.6

Keras 2.2.4

Tensorflow 1.10.0

# Future Work
Due to computational limitations, very little experimentation was able to be done on the architecture of the CNN. The little experimentation
that was performed could not be repeated enough times to yield any statistically significant improvements. Ultimately, we would like to implement
a second convolutional layer with 64 nodes and a kernel size of 3 x 3 with ReLU activation. As in the first layer, this convolutional layer would
be immediately followed by a max pooling layer and a dropout layer. It would also be interesting to increase the number of hidden layers in the ANN.
Currently, we have one fully connected layer. We hypothesize that a second hidden layer with 32 nodes and ReLU activation would improve the CNN's
performance.
