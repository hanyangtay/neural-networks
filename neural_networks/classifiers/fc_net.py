import numpy as np

from neural_networks.layers import *
from neural_networks.layer_utils import *
from neural_networks.layer_utils import affine_batch_relu_forward
from neural_networks.layer_utils import affine_batch_relu_backward



class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    
    dims = [input_dim] + hidden_dims + [num_classes]

    # initialise all parameters (weight, bias, gamma, beta)
    for i in range(len(dims)-1):
      w = 'W' + str(i+1)
      b = 'b' + str(i+1)
      self.params[w] = np.random.randn(dims[i], dims[i+1])*weight_scale
      self.params[b] = np.zeros(dims[i+1])
      
    if self.use_batchnorm:
      for i in range(len(dims)-2):
        #no gamma and beta for last layer
        gamma = 'gamma' + str(i+1)
        beta = 'beta' + str(i+1)
        self.params[gamma] = np.ones(dims[i+1])
        self.params[beta] = np.zeros(dims[i+1])
        
   

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    
    ### forward pass ###
    L = self.num_layers
    past_caches = [0 for i in range(L)]
    
    if self.use_dropout:
      dropout_caches = [0 for i in range(L)]
    
    out = X
    if self.use_batchnorm:
      for i in range(L-1):

        out, past_caches[i] = affine_batch_relu_forward(out, self.params['W' + str(i+1)],
                                                 self.params['b' + str(i+1)], 
                                                 self.params['gamma' + str(i+1)],
                                                 self.params['beta' + str(i+1)],
                                                 self.bn_params[i])
        if self.use_dropout:
          out, dropout_caches[i] = dropout_forward(out, self.dropout_param)
    else:
      for i in range(L-1):

        out, past_caches[i] = affine_relu_forward(out, self.params['W' + str(i+1)],
                                                 self.params['b' + str(i+1)])
        if self.use_dropout:
          out, dropout_caches[i] = dropout_forward(out, self.dropout_param)
      
    scores, past_caches[L-1] = affine_forward(out, self.params['W' + str(L)],
                                             self.params['b' + str(L)])
    
    ### backpropagation ###

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    
    douts = [0 for i in range(L+1)] #account for the final dout 
    loss_l2 = 0
    
    loss, douts[L] = softmax_loss(scores, y)
    for i in range(L-1): 
      W = 'W' + str(i+1)
      loss_l2 += np.sum(self.params[W]*self.params[W])
    loss_l2 *= 0.5 * self.reg
    loss += loss_l2
    
    W_final = 'W'+str(L)
    b_final = 'b'+str(L)
    douts[L-1], grads[W_final], grads[b_final] = affine_backward(douts[L], past_caches[L-1])
    grads[W_final] += self.reg * self.params[W_final]
    
    if self.use_batchnorm:
      for i in range(L-1):
          ind = L-1-i
          W = 'W'+str(ind)
          b = 'b'+str(ind)
          gamma = 'gamma' + str(ind)
          beta = 'beta' + str(ind)
          
          if self.use_dropout:
            douts[-i-2] = dropout_backward(douts[-i-2], dropout_caches[-i-2])

          douts[-i-3], grads[W], grads[b], grads[gamma], grads[beta] = affine_batch_relu_backward(douts[-i-2], past_caches[-i-2])
          grads[W] += self.reg * self.params[W]

    else:
      for i in range(L-1):
        ind = L-1-i
        W = 'W'+str(ind)
        b = 'b'+str(ind)
        
        if self.use_dropout:
            douts[-i-2] = dropout_backward(douts[-i-2], dropout_caches[-i-2])

        douts[-i-3], grads[W], grads[b] = affine_relu_backward(douts[-i-2], past_caches[-i-2])
        grads[W] += self.reg * self.params[W]

    return loss, grads
