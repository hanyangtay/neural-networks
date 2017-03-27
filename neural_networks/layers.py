import numpy as np
"""
        Backpropagation Notes
        Aim: Change weights and biases to reduce cost
        
        delta   = error of neuron
                = dCost wrt weighted input, z
                  (if this is small, cost won't change much, 
                  and is near optimal)
                  
        *******1*******  
        delta of output neuron
        delta   = dCost wrt activation, a * da wrt z (chain rule)
                = dCost wrt a * activation_function_deriv(z)
                
        *******2******* 
        delta of hidden layer neuron
        delta   = dCost wrt z_nextlayer * dz_nextlayer wrt z 
                = delta_nextlayer * dz_nextlayer wrt z
                = w_nextlayer * activation_function_deriv(z) * delta_nextlayer
                
        Note that z_nextlayer = w_nextlayer * a + b_nextlayer,
        differentiating it with chain rule yields
        
        *******3******* 
        delta_b
        dCost wrt bias  = dCost wrt bias * dbias wrt z
                        = dCost wrt z
                        = delta
        
        Note that z = w*x + bias, so bias = z - w*x, dbias wrt z = 1      
        
        *******4*******  
        delta_w
        dCost wrt weights   = dCost wrt z * dz wrt weights
                            = delta * a_in
        
        Note that z = weights * a_in + b
                  
"""


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None

  N = x.shape[0]
  D = np.prod(x.shape[1:])
  
  out = x.reshape((N, D)).dot(w) + b
  cache = (x, w, b)
  
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x.reshape(N, D).T.dot(dout)
  db = dout.T.sum(axis=1)

  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """

  out= np.maximum(0, x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = dout * (x>=0)

  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
  
    sample_mean = np.mean(x, axis=0)
    sample_var = np.std(x, axis=0) **2

    #exponentially decaying running mean and var
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    
    #normalise data
    x_norm = x - sample_mean
    std = np.sqrt(sample_var+eps)
    out_x = x_norm/std
    
    out = out_x * gamma + beta
    cache = (x_norm, std, out_x, gamma)
    
  elif mode == 'test':
    
    #normalise data
    x_norm = x - running_mean
    std = np.sqrt(running_var+eps)
    out_x = x_norm / std
    
    out = out_x * gamma + beta
    cache = (x_norm, std, out_x, gamma)

  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  x_norm, std, out_x, gamma= cache

  N = dout.shape[0]
  
  dbeta = dout.T.sum(axis=1)
  dgamma = np.sum(out_x * dout, axis=0)
  
  dxmu = dout * gamma / std
  
  #used computational graph to derive local gradients
  dinvvar = np.sum((x_norm)*gamma * dout, axis=0)* -1 / std**2 * 0.5 / std / N * 2 * (x_norm)
  dxmu += dinvvar
  
  dx = dxmu
  
  dmu = -np.sum(dxmu, axis=0) / N
  dx += dmu

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  mask = None
  
  if mode == 'train':
    # divide p is for inverted dropout to improve test-time performance
    mask = (np.random.rand(*x.shape)<p) /p
    out = x * mask

  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']

  if mode == 'train':
    dx = dout * mask

  elif mode == 'test':
    dx = dout
    
  return dx


def conv_forward(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  pad = conv_param['pad']
  stride = conv_param['stride']
  
  N, _, H, W = x.shape
  F, _, HH, WW = w.shape
  H_act = 1 + (H + 2*pad - HH) / stride
  W_act = 1 + (W + 2*pad - WW) / stride
  
  x_pad = np.pad(x, [(0,), (0,), (pad,), (pad,)], 'constant')
  out = np.zeros([N, F, H_act, W_act])
  
  for n in range(N):
    for f in range(F):
      for i in range(H_act):
        for j in range(W_act):
          row = i * stride
          col = j * stride
          out[n, f, i, j] = np.sum(x_pad[n, :, row: row+HH, col:col+WW]*w[f])+b[f]
          
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """

  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']

  
  N, _, H, W = x.shape
  F, _, HH, WW = w.shape
  H_act = 1 + (H + 2*pad - HH) / stride
  W_act = 1 + (W + 2*pad - WW) / stride
  
  x_pad = np.pad(x, [(0,), (0,), (pad,), (pad,)], 'constant')
  
  dx = np.zeros_like(x_pad)
  dw = np.zeros_like(w)

  db = np.sum(dout, axis=(0, 2, 3))
  
  for n in range(N):
    for f in range(F):
      for i in range(H_act):
        for j in range(W_act):
          row = i * stride
          col = j * stride
          for k in range(HH):
            for l in range(WW):
              dw[f, :, k, l] +=  x_pad[n, :, row+k, col+l] * dout[n, f, i, j]
              dx[n, :, row+k, col+l] += dout[n,f,i,j] * w[f,:,k,l]
              
  dx = dx[:,:,pad:-pad, pad:-pad]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """

  N, C, H, W = x.shape
  pool_h = pool_param['pool_height']
  pool_w =pool_param['pool_width']
  stride = pool_param['stride']
  
  H2 = 1 + (H - pool_h) / stride
  W2 = 1 + (W - pool_w) / stride
  
  out = np.zeros([N, C, H2, W2])
  
  for n in range(N):
    for h in range(H2):
      for w in range(W2):
        row = h * stride
        col = w * stride
        out[n, :, h, w] = np.max(x[n, :, row:row+pool_h, col:col+pool_w], axis=(1,2))

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  
  N, C, H, W = x.shape
  pool_h = pool_param['pool_height']
  pool_w =pool_param['pool_width']
  stride = pool_param['stride']
  
  H2 = 1 + (H - pool_h) / stride
  W2 = 1 + (W - pool_w) / stride
  
  dx = np.zeros_like(x)
  
  out = np.zeros([N, C, H2, W2])
  for n in range(N):
    for c in range(C):
      for h in range(H2):
        for w in range(W2):
          row = h * stride
          col = w * stride
          x_pool = x[n, c, row:row+pool_h, col:col+pool_w]
          max_val = np.max(x_pool)
          x_mask = np.zeros_like(x_pool)
          x_mask[x_pool == max_val] = 1
          dx[n, c, row:row+pool_h, col:col+pool_w] += dout[n, c, h, w] * x_mask

  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  N, C, H, W = x.shape
  cache = [0 for i in range(C)]
  x = np.reshape(x, (N, C, H*W))
  out = np.zeros_like(x)
  
  for c in range(C):
    out[:,c], cache[c] = batchnorm_forward(x[:,c], gamma[c], beta[c], bn_param)
  out = np.reshape(out, (N, C, H, W))

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """

  N, C, H, W = dout.shape
  dout = np.reshape(dout, (N, C, H*W))
  
  dx = np.zeros_like(dout)
  dgamma = np.zeros(C)
  dbeta=np.zeros(C)
  
  for c in range(C):
    dx[:,c], dgamma_temp, dbeta_temp = batchnorm_backward(dout[:,c], cache[c])
    dgamma[c] = np.sum(dgamma_temp)
    dbeta[c] = np.sum(dbeta_temp)
    
  dx = np.reshape(dx, (N, C, H, W))

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos

  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
