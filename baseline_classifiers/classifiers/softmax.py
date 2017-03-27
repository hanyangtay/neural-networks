import numpy as np

def softmax_loss(W, X, y, reg):
  """
  Softmax loss function
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  
  f = X.dot(W)
  f -= np.max(f) #shift the values so that the highest exponent is zero
  exp_f = np.exp(f)
  sum_f = np.sum(exp_f, axis=1, keepdims=True)
  loss = np.sum(-f[np.arange(num_train),y]) + np.sum(np.log(sum_f))
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  
  softmax = exp_f / sum_f
  softmax[np.arange(num_train), y] -= 1
  dW= X.T.dot(softmax)
  
  dW /= num_train
  dW += reg * W

  return loss, dW

