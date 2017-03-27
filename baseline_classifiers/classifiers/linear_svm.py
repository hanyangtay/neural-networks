import numpy as np

def svm_loss(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape)
  num_train = X.shape[0]
  
  scores = X.dot(W)
  correct_scores = scores[np.arange(num_train), y]
  margin = scores - correct_scores[:, None] + 1 #delta = 1

  margin[np.arange(num_train), y] = 0
  indicator = np.zeros(margin.shape)
  indicator[margin > 0] = 1
  
  loss = np.sum(indicator * margin) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #calculate dW

  counter = np.sum(indicator, axis=1)
  indicator[np.arange(num_train), y] = -counter
  dW = X.T.dot(indicator)

  dW /= num_train
  dW += reg * W

  return loss, dW
