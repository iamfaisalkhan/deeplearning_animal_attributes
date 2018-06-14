# Predict The Attributes of the Animals

Our submission for the HackerEarth challenge for predicting the attributes of animals from the image. The data consists of 

Here we explain the deep learning architectures used along with the type of augmentation used for our solution. This solution achieved a f-score of 0.97901 on the online data. 

# Data Augmentation

We only used the provided dataset and no external images were added. However, we generated extra training examples by augmenting the existing one. These augmentation helps to train a network on  Here is a brief summary of all the augmentation used in building the final model. 

### Brightness

A random factor of brightness was added to each image using the HSV color space. This can potentially allow us to generalize beyond the seen data and handle shadows and lightning conditions in the tracks. 

### Affine Transformation

A small set of affine transformations (translation, shear, rotation) were applied to each 

[image1]: ./augmented_data.png "Augmentation applied to the data"


![alt text][image1]

#### Flip

We randomly flipped images around the y axis. 

### Zoom

### Channel Shift

# Architecture

We experimented with following deep neural network architectures. 

- MobileNet
- Resnet50
- DenseNet121, DenseNet169, DenseNet201
- Ensemble of (DenseNet121 + DenseNet169, DenseNet201)
- Ensemble of (DenseNet201 + Resnet50)

Our final solution is based on the ensemble of the three DenseNet architectures below. These architectures were initialized with imagenet data and then trained on the animal data. We first fine tuned the last layer to make sure that, in the initial epochs, the gradient updates from random weight doesn't adversely affect the weights on previous layers. For ensemble, we averaged the output of the sigmoid activation layer of the models involved. 

The ensemble provided only a very small performance boost (~0.5 pts)and can potentially be avoided in favor a single DenseNet201 model. We used an input image resolution of (224 by 224) for all networks. 


# Training

We split the training data into roughly 80-20 splits. All the trainings were done with AdamOptimizer with initial leanring rate of 1e-4. The training rate was reduce few times manually as the validation loss plateaued. Early stopping with best model saving was also used from Keras to avoid overfitting the model to the training data. 

# Implementation

These network were trained using Keras with Tensorflow backend. We used the already implemented models form Keras.Applications API. Additionally, these model were initialized with imagenet data without the top layers. The code is contained in individual jupyter notebooks.


# Acknowledgment.

Thanks to Dhaval Mayatra for providing an excellent start kit (https://github.com/dhavalmj007/DL3-transfer-learning-starter-kit). 
