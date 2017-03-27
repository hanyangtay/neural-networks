import numpy as np

from neural_networks.layers import *
from neural_networks.fast_layers import *
from neural_networks.layer_utils import *

class OptimalConvNet(object):
    """
    A L-layer convolutional network with the following architecture:
    [conv-relu-pool]*M - [affine - relu]*N - affine - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 32], filter_size=3,
                 hidden_dims=[100, 100], num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: List of size  Nbconv+1 with the number of filters
        to use in each convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dims: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}

        self.filter_size = filter_size
        self.M = len(num_filters)  # Number of weights
        self.N = len(hidden_dims)  # Number of conv/relu/pool blocks

        # Size of the input
        C, H, W = input_dim
        stride_conv = 1  # stride

        # Initialize the weight for the conv layers
        F = [C] + num_filters
        for i in xrange(self.M):
            ind = i + 1

            self.params['W'+str(ind)] = weight_scale * np.random.randn(F[i + 1], 
                                                                       F[i], 
                                                                       self.filter_size, 
                                                                       self.filter_size)
            self.params['b' + str(ind)] = np.zeros(F[i + 1])
            
            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                            'running_mean': np.zeros(F[i + 1]),
                            'running_var': np.zeros(F[i + 1])}
                self.params['gamma' + str(ind)] = np.ones(F[i + 1])
                self.params['beta' + str(ind)] = np.zeros(F[i + 1])
                self.bn_params['bn_param' + str(ind)] = bn_param

        # Initialize the weights for the affine-relu layers, including last layer
  
        # Size of the last conv layer activation
        Hconv, Wconv = self.Size_Conv(
            stride_conv, self.filter_size, H, W, self.M)
      
        dims = [Hconv * Wconv * F[-1]] + hidden_dims + [num_classes]
        
        for i in range(self.N):
            ind = self.M + i + 1
            self.params['W'+str(ind)] = weight_scale * np.random.randn(dims[i], dims[i + 1])
            self.params['b'+str(ind)] = np.zeros(dims[i + 1])

            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                            'running_mean': np.zeros(dims[i + 1]),
                            'running_var': np.zeros(dims[i + 1])}
                self.params['gamma' + str(ind)] = np.ones(dims[i + 1])
                self.params['beta' + str(ind)] = np.zeros(dims[i + 1])
                self.bn_params['bn_param' + str(ind)] = bn_param

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def Size_Conv(self, stride_conv, filter_size, H, W, Nbconv):
        P = (filter_size - 1) / 2  # padd
        Hc = (H + 2 * P - filter_size) / stride_conv + 1
        Wc = (W + 2 * P - filter_size) / stride_conv + 1
        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1
        if Nbconv == 1:
            return Hp, Wp
        else:
            H = Hp
            W = Wp
            return self.Size_Conv(stride_conv, filter_size, H, W, Nbconv - 1)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        N = X.shape[0]

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        scores = None
        
        ### forward pass ###
        all_layers = [(None, None) for i in range(self.M+self.N+1)]
        
        # Forward pass into the  M * conv layers
        for i in range(self.M):
            ind = i + 1
            W = self.params['W' + str(ind)]
            b = self.params['b' + str(ind)]
            h, _ = all_layers[ind - 1]
            if self.use_batchnorm:
                bn_param = self.bn_params['bn_param' + str(idx)]
                gamma = self.params['gamma' + str(ind)]
                beta = self.params['beta' + str(ind)]
                h, cache_h = conv_bn_relu_pool_forward(
                    h, W, b, conv_param, bn_param, gamma, beta, pool_param)
            else:
                h, cache_h = conv_relu_pool_forward(
                    h, W, b, conv_param, pool_param)
            all_layers[ind] = (h, cache_h)


        # Forward pass into the N * affine-relu layers
        all_layers[self.M] = h.reshape(N, np.product(h.shape[1:])), cache_h
        for i in range(self.N):
            ind = self.M + i + 1
            h, _ = all_layers[ind - 1]
            W = self.params['W' + str(ind)]
            b = self.params['b' + str(ind)]
            if self.use_batchnorm:
                bn_param = self.bn_params['bn_param' + str(idx)]
                gamma = self.params['gamma' + str(ind)]
                beta = self.params['beta' + str(ind)]
                h, cache_h = affine_batch_relu_forward(h, W, b, gamma,
                                                      beta, bn_param)
            else:
                h, cache_h = affine_relu_forward(h, W, b)
            all_layers[ind] = (h, cache_h)

        # Forward pass into the last affine - softmax layer
        ind = self.M + self.N + 1
        W = self.params['W' + str(ind)]
        b = self.params['b' + str(ind)]
        h,_ = all_layers[ind-1]
        h, cache_h = affine_forward(h, w, b)
        all_layers[ind] = (h, cache_h)

        scores = h

        if y is None:
            return scores

        loss, grads = 0, {}
        
        ### Backward pass ###

        douts = [0 for i in range(self.L+self.M+1)] #account for the final dout 
        loss_l2 = 0

        loss, douts[self.L+self.M] = softmax_loss(scores, y)
        for i in range(self.L+self.M): 
          W = 'W' + str(i+1)
          loss_l2 += np.sum(self.params[W]*self.params[W])
        loss_l2 *= 0.5 * self.reg
        loss += loss_l2


        # Backward pass
        # print 'Backward pass'
        # Backprop into the scoring layer
        idx = self.L + self.M + 1
        dh = dscores
        h_cache = blocks['cache_h' + str(idx)]
        dh, dw, db = affine_backward(dh, h_cache)
        blocks['dh' + str(idx - 1)] = dh
        blocks['dW' + str(idx)] = dw
        blocks['db' + str(idx)] = db

        # Backprop into the linear blocks
        for i in range(self.M)[::-1]:
            idx = self.L + i + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if self.use_batchnorm:
                dh, dw, db, dgamma, dbeta = affine_norm_relu_backward(
                    dh, h_cache)
                blocks['dbeta' + str(idx)] = dbeta
                blocks['dgamma' + str(idx)] = dgamma
            else:
                dh, dw, db = affine_relu_backward(dh, h_cache)
            blocks['dh' + str(idx - 1)] = dh
            blocks['dW' + str(idx)] = dw
            blocks['db' + str(idx)] = db

        # Backprop into the conv blocks
        for i in range(self.L)[::-1]:
            idx = i + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if i == max(range(self.L)[::-1]):
                dh = dh.reshape(*blocks['h' + str(idx)].shape)
            if self.use_batchnorm:
                dh, dw, db, dgamma, dbeta = conv_norm_relu_pool_backward(
                    dh, h_cache)
                blocks['dbeta' + str(idx)] = dbeta
                blocks['dgamma' + str(idx)] = dgamma
            else:
                dh, dw, db = conv_relu_pool_backward(dh, h_cache)
            blocks['dh' + str(idx - 1)] = dh
            blocks['dW' + str(idx)] = dw
            blocks['db' + str(idx)] = db

        # w gradients where we add the regulariation term
        list_dw = {key[1:]: val + self.reg * self.params[key[1:]]
                   for key, val in blocks.iteritems() if key[:2] == 'dW'}
        # Paramerters b
        list_db = {key[1:]: val for key, val in blocks.iteritems() if key[:2] ==
                   'db'}
        # Parameters gamma
        list_dgamma = {key[1:]: val for key, val in blocks.iteritems() if key[
            :6] == 'dgamma'}
        # Paramters beta
        list_dbeta = {key[1:]: val for key, val in blocks.iteritems() if key[
            :5] == 'dbeta'}

        grads = {}
        grads.update(list_dw)
        grads.update(list_db)
        grads.update(list_dgamma)
        grads.update(list_dbeta)

        return loss, grads
  
  
  
class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    C, H, W = input_dim
    F = num_filters
    HH = filter_size
    WW = filter_size
    stride = 1
    pad = (filter_size-1)/2
    
    H_act = 1 + (H + 2*pad - HH) / stride
    W_act = 1 + (W + 2*pad - WW) / stride
    
    pool_h = 2
    pool_w = 2
    
    stride_pool = 2

    H2 = 1 + (H_act - pool_h) / stride_pool
    W2 = 1 + (W_act - pool_w) / stride_pool
    
    self.params['W1'] = np.random.randn(F, C, HH, WW)*weight_scale
    self.params['b1'] = np.zeros(F)
    self.params['W2'] = np.random.randn(F*H2*W2, hidden_dim)*weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes)*weight_scale
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    reg = self.reg
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # stores results of forward pass 
    scores = None

    ### Forward Pass ###
    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    N, F, H, W = conv_out.shape
    conv_out.reshape(N, F*H*W)
    fc_out, fc_cache = affine_relu_forward(conv_out, W2, b2)
    scores, cache = affine_forward(fc_out, W3, b3)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    
    ### Backpropagation ###
    loss, dx_3 = softmax_loss(scores, y)
    loss += 0.5 * reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))

    dx_2, grads["W3"], grads["b3"] = affine_backward(dx_3, cache)
    grads["W3"] += reg * W3
    
    dx_1, grads["W2"], grads["b2"] = affine_relu_backward(dx_2, fc_cache)
    grads["W2"] += reg * W2
    dx_1.reshape(N, F, H, W)
    
    _, grads["W1"], grads["b1"] = conv_relu_pool_backward(dx_1, conv_cache)
    grads["W1"] += reg * W1

    
    return loss, grads
  
  
pass
