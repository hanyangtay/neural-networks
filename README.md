![alt text](https://github.com/hanyangtay/hanyang/raw/master/app/assets/images/personal/hy.png "Han Yang")

# Neural Networks

To consolidate my knowledge of deep learning, I trained several neural networks over the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The objective is to train a good image classifier. 

Optimal test accuracy: 84.6%

Tools used: Python, numpy, Cython

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


#### Recurrent neural networks

* LSTM

* Text Generation

* Music Generation


## Credits

The modular design of the neural networks and a huge part of the code, is from the [Stanford CS231n course](http://cs231n.github.io/).

I am deeply grateful to the Stanford team, who generously decided to make the course materials public. As somebody who tried to self-learn neural networks from other online tutorials, I really admire the way the course was structured and the pedagogical approach behind it. It was definitely difficult, but I learned so much. Highly recommended. Feel free to view my solutions [here](https://github.com/hanyangtay/CS231n-answers).

