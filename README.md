# Neural Networks

![alt text](https://github.com/hanyangtay/hanyang/raw/master/app/assets/images/personal/hy.png "Han Yang")

To consolidate my knowledge of deep learning from the [Stanford CS231n course](http://cs231n.github.io/), I trained several neural networks over the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The objective is to train a good image classifier. I split the dataset into a training set (49000 images), a validation set (1000 images) and a final test set (1000 images).

These neural networks were trained overnight on my Macbook Pro. Limited by computing power, I was not able to fine-tune my hyperparameters to achieve the best possible result, or to train deep architectures efficiently. Using heuristics and seeing the trend of loss, training accuracy and validation accuracy over the training periods, I manually tuned the learning rates and regularisation rates. 

Tools used: Python, numpy, Cython, Jupyter

## Best ConvNet Model

The best single model is a CNN which achieved a **79.8% test accuracy** over the 1000 images. Its architecture is relatively simple.

There is a conv composite layer comprising of a convolutional layer that uses 128 3x3 filters, spatial batchnormalisation layer, ReLU layer and max pooling layer. There is another fully connected layer comprising of a simple affine transformation layer and a ReLU activation layer. This CNN has 3 conv composite layers, 2 fully connected layers and a final softmax loss layer. In summary, this is the architecture:

(conv - spatial batchnorm - relu - max pool) x 3 - (affine - relu) x 2 - softmax


Using an ensemble of 3 CNN models with varying architectures, I managed to boost test accuracy by 4%. 

**Optimal test accuracy: 84.6%**

Read the following notebooks for more information: 

[Linear Classifiers](https://github.com/hanyangtay/neural-networks/blob/master/Baseline%20Classifiers.ipynb), [Fully Connected Net](https://github.com/hanyangtay/neural-networks/blob/master/Fully%20Connected%20Net.ipynb), [Convolutional Neural Network](https://github.com/hanyangtay/neural-networks/blob/master/CNN.ipynb)

## Types of ML classifiers

#### Linear Classifiers

Trained linear classifiers to act as a baseline performance test.

* [k-nearest Neighbour](https://github.com/hanyangtay/neural-networks/blob/master/Baseline%20Classifiers.ipynb) (27.8%)

* [Multiclass Support Vector Machine](https://github.com/hanyangtay/neural-networks/blob/master/Baseline%20Classifiers.ipynb) (37.8%)

* [Softmax Classifier](https://github.com/hanyangtay/neural-networks/blob/master/Baseline%20Classifiers.ipynb) (37.4%)


#### Artifical neural networks

Trained ANNs and CNNs.

* [Convolutional Neural Network ](https://github.com/hanyangtay/neural-networks/blob/master/CNN.ipynb) (84.6%)

* [Fully Connected Net](https://github.com/hanyangtay/neural-networks/blob/master/Fully%20Connected%20Net.ipynb) (53.6%)





## Future Direction:

* Saving trained models using cPickle

#### Refining artificial neural networks

* Nestarov momentum

* Xavier Initialisation

* Adadelta

* More complex ensembles

This ensemble of 3 CNN models simply added their individual scores together to predict the labels of the images.

Unfortunately, my computer crashed before I could attempt to save my models, but I postulate that I'd' be able to increase the test accuracy by training a fully connected net over the final scores of each individual model.


#### Recurrent neural networks

* LSTM

* Text Generation (inspired by [Andrej Karpathy's character RNN](https://github.com/karpathy/char-rnn))

* Music Generation


## Credits

The modular design of the neural networks and a huge part of the code, is from the [Stanford CS231n course](http://cs231n.github.io/).

I am deeply grateful to the Stanford team, who generously decided to make the course materials public. As somebody who tried to self-learn neural networks from other online tutorials, I really admire the way the course was structured and the pedagogical approach behind it. It was definitely difficult, but I learned so much. Highly recommended. Feel free to view my solutions [here](https://github.com/hanyangtay/CS231n-answers).

