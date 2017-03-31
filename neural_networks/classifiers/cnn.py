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
        self.conv_params = {'stride': 1, 'pad': (filter_size - 1) / 2}
        self.pool_params = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        self.bn_params = {}

        self.filter_size = filter_size
        self.M = len(num_filters)  # Number of weights
        self.N = len(hidden_dims)  # Number of conv/relu/pool blocks

        ### Weight initialisation ###
        
        # Size of the input
        C, H, W = input_dim
        stride_conv = self.conv_params['stride']  # stride

        # Initialize the weight for the conv layers
        F = [C] + num_filters
        for i in range(self.M):
            ind = i + 1

            # dimension of weight (F, C, H, W)
            # number of filters become the number of channels in the
            # next layer (refer to slides again)
            self.params['W'+str(ind)] = weight_scale * \
                                        np.random.randn(F[i + 1], F[i], 
                                                        self.filter_size, 
                                                        self.filter_size)
            # dimension of bias (F)
            self.params['b'+str(ind)] = np.zeros(F[i + 1])
            
            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                            'running_mean': np.zeros(F[i + 1]),
                            'running_var': np.zeros(F[i + 1])}
                self.params['gamma' + str(ind)] = np.ones(F[i + 1])
                self.params['beta' + str(ind)] = np.zeros(F[i + 1])
                self.bn_params['bn_param' + str(ind)] = bn_param

        # Initialize the weights for the affine-relu layers, including last layer
  
        # Size of the last conv layer activation
        Hconv, Wconv = self.size_conv(self.filter_size, H, W, self.M)
      
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

        ind = self.M + self.N + 1
        self.params['W'+str(ind)] = weight_scale * np.random.randn(dims[-2], dims[-1])
        self.params['b'+str(ind)] = np.zeros(dims[-1])        
                
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def size_conv(self, filter_size, H, W, n_conv):
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        
        HH = filter_size #filter height
        WW = filter_size # filter width
        
        H_act = 1 + (H + 2*pad - HH) / stride
        W_act = 1 + (W + 2*pad - WW) / stride
        
        pool_h = self.pool_params['pool_height']
        pool_w = self.pool_params['pool_width']
        stride_pool = self.pool_params['stride']
        
        H2 = 1 + (H_act - pool_h) / stride_pool
        W2 = 1 + (W_act - pool_w) / stride_pool
        
        # return dimensions if it's the last conv layer
        if n_conv == 1: 
            return H2, W2
        # "forward pass" into next conv layer
        else: 
            return self.size_conv(filter_size, H2, W2, n_conv-1)


    
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        M = self.M # number of conv layers
        N = self.N # number of linear layers

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size
        conv_param = self.conv_params
        pool_param = self.pool_params
        
        # pass pool_param to the forward pass for the max-pooling layer
        

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode
                
        ### forward pass ###
        
        #num layers = M+N+1(softmax layer)
        past_caches = [None for i in range(M+N+1)]
        out = X
        # Forward pass into the  M * conv layers
        for i in range(M):
            ind = i + 1
            W = self.params['W' + str(ind)]
            b = self.params['b' + str(ind)]
            if self.use_batchnorm:
                bn_param = self.bn_params['bn_param' + str(ind)]
                gamma = self.params['gamma' + str(ind)]
                beta = self.params['beta' + str(ind)]
                out, past_caches[i] = conv_bn_relu_pool_forward(
                    out, W, b, conv_param, bn_param, gamma, beta, pool_param)
            else:
                out, past_caches[i] = conv_relu_pool_forward(
                    out, W, b, conv_param, pool_param)

                

        # Forward pass into the N * affine-relu layers
        N_c, F, H, W = out.shape
        original_shape = out.shape
        out = out.reshape(N_c, F*H*W)

        for i in range(M, M+N):
            ind = i + 1
            W = self.params['W' + str(ind)]
            b = self.params['b' + str(ind)]
            if self.use_batchnorm:
                bn_param = self.bn_params['bn_param' + str(ind)]
                gamma = self.params['gamma' + str(ind)]
                beta = self.params['beta' + str(ind)]
                out, past_caches[i] = affine_batch_relu_forward(out, W, b, gamma,
                                                      beta, bn_param)
            else:
                out, past_caches[i] = affine_relu_forward(out, W, b)

        # Forward pass into the last affine - softmax layer
        ind = M+N+1
        W = self.params['W' + str(ind)]
        b = self.params['b' + str(ind)]
        scores, past_caches[M+N] = affine_forward(out, W, b)

        if y is None:
            return scores

        
        
        ### Loss Calculation ###
        
        loss, grads = 0, {}
        loss_l2 = 0
        loss, dx = softmax_loss(scores, y)
        for i in range(M+N+1): 
          W = 'W' + str(i+1)
          loss_l2 += np.sum(self.params[W]*self.params[W])
        loss_l2 *= 0.5 * self.reg
        loss += loss_l2

        ### Backpropagation ###        
        
        W = "W" + str(M+N+1)
        b = "b" + str(M+N+1)
        dx, grads[W], grads[b] = affine_backward(dx, past_caches[-1])
        grads[W] += self.reg * self.params[W]

        
        if self.use_batchnorm:
          for i in range(N):
            ind = M+N - i
            W = "W" + str(ind)
            b = "b" + str(ind)
            gamma = 'gamma' + str(ind)
            beta = 'beta' + str(ind)
            
            dx, grads[W], grads[b], grads[gamma], grads[beta] = \
              affine_batch_relu_backward(dx, past_caches[-i-2])
              
            grads[W] += self.reg * self.params[W]
          
          dx = dx.reshape(original_shape)
          
          for i in range(M):
            ind = M - i
            W = "W" + str(ind)
            b = "b" + str(ind)
            gamma = 'gamma' + str(ind)
            beta = 'beta' + str(ind)
            dx, grads[W], grads[b], grads[gamma], grads[beta] = \
              conv_bn_relu_pool_backward(dx, past_caches[-i-N-2])
            grads[W] += self.reg * self.params[W]
            
        else:  
          for i in range(N):
            ind = M+N - i
            W = "W" + str(ind)
            b = "b" + str(ind)
            
            dx, grads[W], grads[b] = affine_relu_backward(dx, past_caches[-i-2])
            grads[W] += self.reg * self.params[W]
          
          for i in range(M):
            ind = M - i
            W = "W" + str(ind)
            b = "b" + str(ind)
            dx, grads[W], grads[b] = conv_relu_pool_backward(dx, 
                                                             past_caches[-i-N-2])
            grads[W] += self.reg * self.params[W]

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
    

    ### Initialise weights ###

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
      print scores.shape
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
