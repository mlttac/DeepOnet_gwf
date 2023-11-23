# -*- coding: utf-8 -*-

"""
Inspired by the DeepONet architecture for advection problems as demonstrated in:
https://github.com/PredictiveIntelligenceLab/ImprovedDeepONets/blob/main/Advection/PI_DeepONet_advection.ipynb

Author: Maria Luisa Taccari (cnmlt@leeds.ac.uk)

"""

import jax
import jax.numpy as np
import optax
from jax import random, grad, jit, random, vmap, hessian, lax
from jax.nn import relu, elu, tanh
from jax.flatten_util import ravel_pytree
from jax.example_libraries import stax, optimizers
from flax import linen
from jax.example_libraries.stax import (
    Conv, Flatten, AvgPool, MaxPool, ConvTranspose,
    Dense, serial, Relu, Sigmoid, Gelu, Tanh, BatchNorm, LeakyRelu, elementwise
)

from typing import Sequence, Any
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
import timeit
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
from typing import Tuple, Any, Sequence


# Data generator class for neural network training
class DataGenerator(data.Dataset):
    """
    Data generator for batching data in neural network training.
    """

    def __init__(self, u, y, s, u2=None, batch_size=64, rng_key=random.PRNGKey(1234), rng_key2=random.PRNGKey(11)):
        self.u = u
        self.y = y
        self.s = s
        self.u2 = u2
        self.N = u.shape[0]
        self.P = y.shape[0] // self.N
        self.batch_size = batch_size
        self.key = rng_key
        self.key2 = rng_key2

    def __getitem__(self, index):
        self.key, subkey = random.split(self.key)
        self.key2, subkey2 = random.split(self.key2)
        inputs, outputs = self.__data_generation(subkey, subkey2)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, key2):
        # Generates data containing batch_size samples
        idx = random.choice(key, self.N, (self.batch_size,), replace=True)
        idx_2 = random.choice(key2, self.P, (self.batch_size,), replace=True)
        s = self.s[self.P * idx + idx_2, :]
        y = self.y[self.P * idx + idx_2, :]
        u = self.u[idx, :]

        if self.u2 is None:
            inputs = (u, y)
        else:
            u2 = self.u2[idx, :]
            inputs = (u, u2, y)

        outputs = s
        return inputs, outputs

# Custom activation functions
def hat(x: np.ndarray) -> np.ndarray:
    """'Mexican hat' activation function."""
    return 2 * np.maximum(0, x) - 4 * np.maximum(0, x - 0.5) + 2 * np.maximum(0, x - 1)

def hat_curved(x: np.ndarray) -> np.ndarray:
    """Modified 'Mexican hat' activation function with curved shape."""
    return (np.maximum(0, x)**2 - 3 * np.maximum(0, x - 1)**2 +
            3 * np.maximum(0, x - 2)**2 - np.maximum(0, x - 3)**2)

def s_relu(x: np.ndarray) -> np.ndarray:
    """Symmetric ReLU activation function."""
    return np.maximum(0, -x + 1) * np.maximum(0, x)

# Neural network architectures for Encoder and Decoder
def Encoder() -> Tuple[Any, Any]:
    """Create an Encoder neural network."""
    return stax.serial(
        Conv(16, (3, 3), (1, 1), padding="SAME"),
        BatchNorm(), Relu,
        Conv(32, (3, 3), (1, 1), padding="SAME"),
        BatchNorm(), Relu,
        Conv(64, (3, 3), (1, 1), padding="SAME")
    )

def Decoder() -> Tuple[Any, Any]:
    """Create a Decoder neural network."""
    return stax.serial(
        ConvTranspose(64, (3, 3), (1, 1), padding="SAME"),
        BatchNorm(), Relu,
        ConvTranspose(32, (3, 3), (1, 1), padding="SAME"),
        BatchNorm(), Relu,
        ConvTranspose(1, (3, 3), (1, 1), padding="SAME"),
        Relu, Flatten,
        Dense(128), Relu, Dense(100)
    )

class CNN(linen.Module):
    """Custom CNN module using Flax linen."""
    @linen.compact
    def __call__(self, x, activation=linen.relu):
        x = linen.Conv(features=8, kernel_size=(3, 3), padding='SAME')(x)
        x = activation(x)
        x = linen.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = activation(x)
        x = linen.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = linen.Dense(features=128)(x)
        x = linen.tanh(x)
        x = linen.Dense(features=100)(x)
        return x

class CNN_branch(linen.Module):
    """Custom CNN module with branching architecture."""
    nw_last: Sequence[int]
    feat_cnn: Sequence[int]
    ks: Sequence[int]

    @linen.compact
    def __call__(self, x, activation=linen.relu):
        for i in range(len(self.feat_cnn)):
            x = linen.Conv(features=self.feat_cnn[i], kernel_size=(self.ks, self.ks), padding='SAME')(x)
            x = activation(x)
            x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = linen.Dense(features=256)(x)
        x = linen.tanh(x)
        x = linen.Dense(features=512)(x)
        x = linen.tanh(x)
        x = linen.Dense(features=self.nw_last)(x)
        return x



class CNN_trunk_twined(linen.Module):
    """
    Custom CNN Module with a twined architecture for the trunk.
    """
    nw_last: Sequence[int]
    feat_cnn: Sequence[int]
    ks: Sequence[int]

    @linen.compact
    def __call__(self, x, y, activation=linen.relu):
        # Iterate through CNN features to build the twined structure
        for i in range(len(self.feat_cnn)):
            x = linen.Conv(features=self.feat_cnn[i], kernel_size=(self.ks, self.ks), padding='SAME')(x)
            x = activation(x)
            x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

            # Transform y and combine it with x using a learned alpha
            y_transformed = linen.Dense(features=x.shape[1] * x.shape[2] * self.feat_cnn[i])(y)
            y_transformed = activation(y_transformed)
            x_alpha = linen.Dense(features=x.shape[1] * x.shape[2] * self.feat_cnn[i])(x.reshape((x.shape[0], -1)))
            x_alpha = linen.sigmoid(x_alpha)
            y = y_transformed * x_alpha
            x = np.concatenate((x, y.reshape(1, x.shape[1], x.shape[2], self.feat_cnn[i])), axis=-1)

        x = x.reshape((x.shape[0], -1))
        x = linen.Dense(features=256)(x)
        x = activation(x)
        x = linen.Dense(features=512)(x)
        x = activation(x)
        x = linen.Dense(features=self.nw_last)(x)

        y = linen.Dense(features=self.nw_last)(y)
        combined_output = np.sum(x * y)
        return combined_output


class CNN_MLP_twined(linen.Module):
    """
    Custom CNN-MLP Module with twined architecture.
    """
    
    @linen.compact
    def __call__(self, x, x2, y=None):
        # Convolutional layers
        feat1, feat2, feat3 = 16, 16, 64
        feat1_x2, feat2_x2 = 4, 4

        x = linen.Conv(features=feat1, kernel_size=(5, 5), padding='same')(x)
        x = linen.relu(x)
        x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x2 = linen.Dense(features=16 * 16 * feat1_x2)(x2)
        x2 = linen.relu(x2)
        x = np.concatenate((x, x2.reshape(x2.shape[0], 16, 16, feat1_x2)), axis=-1)

        # Additional layers and operations
        x = linen.Conv(features=feat2, kernel_size=(5, 5), padding='same')(x)
        x = linen.relu(x)
        x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        if y is not None:
               y = linen.Dense(features=128)(y)
               y = linen.leaky_relu(y)    
               y = np.append(y, x2)
               y = linen.Dense(features=128)(y)
               y = linen.leaky_relu(y) 
               
        x2 = linen.Dense(features=8*8*feat2_x2)(x2)
        x2 = linen.relu(x2)
        x = np.concatenate((x, x2.reshape(x2.shape[0], 8, 8, feat2_x2)), axis = -1)
        x = linen.Conv(features=feat3, kernel_size=(5, 5),strides=(1,1))(x) #(-1, 8, 8, 64)
        x = linen.relu(x)   
        
        if y is not None:
               y = np.append(y, x2)
               y = linen.Dense(features=128)(y)
               y = linen.leaky_relu(y) 

        # Finalizing the network output
        x = x.reshape((x.shape[0], -1))
        x = linen.Dense(features=256)(x)
        x = linen.tanh(x)
        x = linen.Dense(features=512)(x)
        x = linen.tanh(x)
        x = linen.Dense(features=150)(x)
        x2 = linen.Dense(features=150)(x2)
        x2 = linen.relu(x2)

        # Combine x and x2
        if y is not None:
            y = np.append(y, x2)
            y = linen.Dense(features=150)(y)
            combined_output = np.sum(x * y)
            return combined_output
        return x

class MLP_Flax(linen.Module):
  features: Sequence[int]   
  @linen.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = linen.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = linen.tanh(x)
    return x

def MLP(layers, activation=relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = np.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply


def modified_MLP(layers, activation=relu):
  def xavier_init(key, d_in, d_out):
      glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
      W = glorot_stddev * random.normal(key, (d_in, d_out))
      b = np.zeros(d_out)
      return W, b

  def init(rng_key):
      U1, b1 =  xavier_init(random.PRNGKey(12345), layers[0], layers[1])
      U2, b2 =  xavier_init(random.PRNGKey(54321), layers[0], layers[1])
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          W, b = xavier_init(k1, d_in, d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return (params, U1, b1, U2, b2) 

  def apply(params, inputs):
      params, U1, b1, U2, b2 = params
      U = activation(np.dot(inputs, U1) + b1)
      V = activation(np.dot(inputs, U2) + b2)
      for W, b in params[:-1]:
          outputs = activation(np.dot(inputs, W))
          inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V) 
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply


def modified_deeponet(branch_layers, trunk_layers, activation=relu):

  def xavier_init(key, d_in, d_out):
      glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
      W = glorot_stddev * random.normal(key, (d_in, d_out))
      b = np.zeros(d_out)
      return W, b
  
  def init(rng_key1, rng_key2):
      U1, b1 =  xavier_init(random.PRNGKey(12345), branch_layers[0], branch_layers[1])
      U2, b2 =  xavier_init(random.PRNGKey(54321), trunk_layers[0], trunk_layers[1])
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          W, b = xavier_init(k1, d_in, d_out)
          return W, b
      key1, *keys1 = random.split(rng_key1, len(branch_layers))
      key2, *keys2 = random.split(rng_key2, len(trunk_layers))
      branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
      trunk_params = list(map(init_layer, keys2, trunk_layers[:-1], trunk_layers[1:]))
      return (branch_params, trunk_params, U1, b1, U2, b2)

  def apply(params, u, y):
      branch_params, trunk_params, U1, b1, U2, b2 = params
      U = activation(np.dot(u, U1) + b1)
      V = activation(np.dot(y, U2) + b2)
      for k in range(len(branch_layers)-2):
          W_b, b_b =  branch_params[k]
          W_t, b_t =  trunk_params[k]

          B = activation(np.dot(u, W_b) + b_b)
          T = activation(np.dot(y, W_t) + b_t)

          u = np.multiply(B, U) + np.multiply(1 - B, V) 
          y = np.multiply(T, U) + np.multiply(1 - T, V) 

      W_b, b_b =  branch_params[-1]
      W_t, b_t =  trunk_params[-1]
      B = np.dot(u, W_b) + b_b
      T = np.dot(y, W_t) + b_t
      outputs = np.sum(B * T)
      return outputs

  return init, apply

"""# DeepONet"""

class DeepONet:
    """
    Deep Operator Network (PI-DeepONet) class.
    Implements various architectures for learning operators.
    """

    def __init__(self, arch, weights, branch_layers, trunk_layers, cnn_dim=None, PI=False, lr=1e-3, ks=5, feat_cnn=[16, 16, 16, 64], act_trunk='relu', decoder='multiplication', normalized_loss=False):
        # Initialization of network parameters and architecture
        self.arch = arch
        self.weights = weights
        self.cnn_dim = cnn_dim
        self.PI = PI
        self.normalized_loss = normalized_loss
        self.decoder = decoder
        self.activation_trunk = self.select_activation(act_trunk)

        # Network initialization based on architecture
        params = self.initialize_network(branch_layers, trunk_layers, ks, feat_cnn)

        # Set up optimizer
        self.setup_optimizer(lr, params)
        
        # Logger setup
        self.setup_logging()

    def select_activation(self, activation_name):
        """
        Selects the activation function based on the provided name.
        """
        activation_functions = {
            'relu': relu, 's_relu': s_relu, 'leaky_relu': jax.nn.leaky_relu,
            'elu': elu, 'tanh': tanh, 'hat': hat, 'hat_curved': hat_curved
        }
        return activation_functions.get(activation_name, relu)

    def initialize_network(self, branch_layers, trunk_layers, ks, feat_cnn):
        """
        Initializes the network based on the specified architecture.
        """
        params = None

        if self.arch == 'MLP':
            self.branch_init, self.branch_apply = MLP(branch_layers, activation=relu)
            self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=self.activation_trunk)
            branch_params = self.branch_init(rng_key=random.PRNGKey(1234))
            trunk_params = self.trunk_init(rng_key=random.PRNGKey(4321))
            params = (branch_params, trunk_params)


        if self.arch =='modified_MLP':
            self.branch_init, self.branch_apply = modified_MLP(branch_layers, activation=tanh)
            self.trunk_init, self.trunk_apply = modified_MLP(trunk_layers, activation=self.activation_trunk)
            branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
            trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
            params = (branch_params, trunk_params)
        
        if self.arch =='modified_deeponet':
            self.init, self.apply = modified_deeponet(branch_layers, trunk_layers, activation=tanh)
            params = self.init(rng_key1 = random.PRNGKey(1234), rng_key2 = random.PRNGKey(4321))
        
        if self.arch =='CNN':   
            self.cnn = CNN_branch(nw_last = trunk_layers[-1],  ks = ks , feat_cnn = feat_cnn)
            branch_params = self.cnn.init(random.PRNGKey(0), np.ones((1, 32, 32, self.cnn_dim)), activation=linen.relu )
            self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation= self.activation_trunk ) 
            trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
            if self.decoder != 'non_linear':
                   params = (branch_params, trunk_params)  
            else:
                   self.decoder_init, self.decoder_apply = MLP([300,100,1], activation= self.activation_trunk ) 
                   decoder_params = self.decoder_init(rng_key = random.PRNGKey(4321))
                   params = (branch_params, trunk_params, decoder_params)

        if self.arch =='CNN_MLP' or self.arch == 'CNN_MLP_FNNwell':   
            self.cnn = CNN_branch(nw_last = trunk_layers[-1],  ks = ks , feat_cnn = feat_cnn)
            branch_params1 = self.cnn.init(random.PRNGKey(0), np.ones((1, 32, 32, self.cnn_dim)), activation=linen.relu )
            self.branch_init, self.branch_apply = modified_MLP(branch_layers, activation=jax.nn.leaky_relu) #MLP(branch_layers, activation=relu)
            branch_params2 = self.branch_init(rng_key = random.PRNGKey(1234))                                   
            self.trunk_init, self.trunk_apply = modified_MLP(trunk_layers, activation=self.activation_trunk) #this works well with 1 well! 
            trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
            params = (branch_params1, branch_params2, trunk_params)
          
        if self.arch =='CNN_MLP_twined':
           self.cnn = CNN_MLP_twined()
           branch_params = self.cnn.init(random.PRNGKey(0), np.ones((1, 32, 32, self.cnn_dim)), np.ones((1, branch_layers[0])) )           
           self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=self.activation_trunk)
           trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
           params = (branch_params, trunk_params)
   
        elif self.arch == 'Unet':
            from model.unet import UNet
            self.unet = UNet()
            init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
            branch_params = self.unet.init(init_rngs, np.ones([1, 32, 32, 2]))
            self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=self.activation_trunk)
            trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
            params = (branch_params, trunk_params)

        if self.arch =='CNN_MLP_trunk_twined':   
           self.cnn = CNN_MLP_twined()
           params = self.cnn.init(random.PRNGKey(0), np.ones((1, 32, 32, self.cnn_dim)), np.ones((1, branch_layers[0])), np.ones((1, trunk_layers[0])) )           
                  
        if self.arch =='CNN_trunk_twined':      
           self.cnn = CNN_trunk_twined(nw_last = trunk_layers[-1],  ks = ks , feat_cnn = feat_cnn)
           params = self.cnn.init(random.PRNGKey(0), np.ones((1, 32, 32, self.cnn_dim)),np.ones((1, trunk_layers[0])) )           

        if self.arch =='Encoder_Decoder':   
           self.encoder_init, self.encoder_apply = Encoder()
           self.decoder_init, self.decoder_apply = Decoder()

           # Initialize parameters
           k1, k2 = random.split(random.PRNGKey(0), 2)
           out_dim, encoder_params = self.encoder_init(k1, (-1, 32, 32, 1))
           _, decoder_params = self.decoder_init(k2, out_dim)
           self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=self.activation_trunk)
           trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
           params = (encoder_params, decoder_params, trunk_params)
           
        return params


    def setup_optimizer(self, lr, params):
        """
        Sets up the optimizer with exponential decay.
        """
        lr_schedule = optimizers.exponential_decay(lr, decay_steps=1000, decay_rate=0.9)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_schedule)
        self.opt_state = self.opt_init(params)
        _, self.unravel_params = ravel_pytree(params)

    def setup_logging(self):
        """
        Initializes logging utilities for training and testing.
        """
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_log_test = []
        self.loss_res_log = []



    def operator_net(self, params, u, xx, yy , xx2, yy2, xx3, yy3, u2 = None):

        if xx2 == None:
               y = np.stack([xx, yy]) 
        elif yy2 == None:
              y = np.stack([xx, yy, xx2])
        elif xx3 == None:
               y = np.stack([xx, yy, xx2, yy2])
        else:
               y= np.stack([xx, yy, xx2, yy2,  xx3, yy3])
                            
        if self.arch == 'modified_deeponet':

               outputs = self.apply(params, u, y)
               
        elif self.arch =='CNN_MLP':  
               branch_params1, branch_params2, trunk_params = params
               B1 = self.cnn.apply( branch_params1, u.reshape(1,32,32,self.cnn_dim)) 
               B2 = self.branch_apply(branch_params2, u2)
               B = B1 * B2
               # np.multiply(a,b)
               T = self.trunk_apply(trunk_params, y)
               outputs = np.sum(B * T)
               
        elif self.arch =='CNN_MLP_FNNwell':  
               branch_params1, branch_params2, trunk_params = params
               B = self.cnn.apply( branch_params1, u.reshape(1,32,32,self.cnn_dim)) 
               B2 = self.branch_apply(branch_params2, u2)
               
               # B = np.concatenate((np.squeeze(B1), B2), axis = -1)
               
               # np.multiply(a,b)
               T1= self.trunk_apply(trunk_params, y)
               T = T1 * B2
               # T = np.concatenate((T1, B2), axis = -1)
               outputs = np.sum(B * T)               
        
        elif self.arch =='CNN_MLP_twined':  
               branch_params, trunk_params = params
               B = self.cnn.apply( branch_params, u.reshape(1,32,32,self.cnn_dim), u2.reshape(1,-1)) 
               # B2 = self.mlp.apply(branch_params2, u2)
               # B = B1 * B2
               T = self.trunk_apply(trunk_params, y) #y[:2]
               outputs = np.sum(B * T) 
        elif self.arch =='CNN_MLP_trunk_twined':  

              outputs = self.cnn.apply( params, u.reshape(1,32,32,self.cnn_dim), u2.reshape(1,-1), y.reshape(1, -1)) 
              
        elif self.arch =='CNN_trunk_twined':  
               outputs = self.cnn.apply( params, u.reshape(1,32,32,self.cnn_dim), y.reshape(1, -1)) 

              
        elif self.arch =='Encoder_Decoder':  
               encoder_params, decoder_params, trunk_params = params
               output_encoder = self.encoder_apply(encoder_params, u.reshape(1,32,32,self.cnn_dim))
               B = self.decoder_apply(decoder_params, output_encoder)
               T = self.trunk_apply(trunk_params, y) #y[:2]
               outputs = np.sum(B * T)    
               
        elif self.arch =='Unet':                 
               branch_params, trunk_params = params
               B = self.unet.apply( branch_params, u.reshape(1,32,32,2), mutable=["batch_stats"], rngs={'dropout': jax.random.PRNGKey(2)}) 

               T = self.trunk_apply(trunk_params, y) #y[:2]
               # outputs = np.sum(B * T) 
               outputs = np.sum(B[0].reshape((150,)) *T)

        else:
               if self.decoder != 'non_linear':
                      branch_params, trunk_params = params 
               else:
                      branch_params, trunk_params,decoder_params = params 
                      
               if self.arch == 'CNN':
                       B = self.cnn.apply( branch_params, u.reshape(1,32,32,self.cnn_dim)) 
               else:
                      B = self.branch_apply(branch_params, u)
               T = self.trunk_apply(trunk_params, y)
               
               if self.decoder == 'multiplication':
                      outputs = np.sum(B * T)
               if self.decoder == 'enriched':
                       outputs1 = np.sum(B * T)
                       outputs2 = np.sum(B + T)
                       outputs = outputs1 + outputs2
               if self.decoder == 'non_linear':

                       D = np.concatenate((T, np.squeeze(B)))
                       D_out = self.decoder_apply(decoder_params, D) #y[:2]
                       outputs = np.sum(D_out)
                      
        return outputs

    def KLDivergence(self, x, y):
        """
        Calculates the Kullback-Leibler Divergence, a measure of how one probability distribution diverges from a second, expected probability distribution.
        Args:
            x: Predicted values.
            y: Ground truth values.
        Returns:
            The KL divergence value.
        """
        num_examples = x.shape[0]
        q = x.reshape(num_examples, -1)
        p = y.reshape(num_examples, -1)
        q_norm = q / np.sum(q, axis=1, keepdims=True)
        p_norm = p / np.sum(p, axis=1, keepdims=True)
        eps = 1e-5
        return np.nansum(p_norm * (np.log(p_norm + eps) - np.log(q_norm + eps)))

    def l2_loss(self, x, alpha):
        """
        Computes the L2 loss, a regularizing loss to prevent overfitting.
        Args:
            x: Input data.
            alpha: Regularization parameter.
        Returns:
            The L2 loss value.
        """
        return alpha * np.mean(x ** 2)

    def loss_operator(self, params, batch, u2=None):
        """
        Computes the operator loss.
        Args:
            params: Network parameters.
            batch: Data batch containing inputs and outputs.
            u2: Additional input, if any.
        Returns:
            Loss value.
        """
        inputs, outputs = batch
        u, y = inputs if len(inputs) == 2 else (inputs[0], inputs[2])
        xx, yy = y[:, 0], y[:, 1]
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, xx, yy, u2)
        loss = np.mean((outputs.flatten() - s_pred.flatten())**2) if not self.normalized_loss else np.mean(((outputs.flatten() - s_pred.flatten())**2) / outputs.flatten())
        return loss

    def residual_net(self, params, u, x, y, u2=None):
        """
        Defines the residual network.
        Args:
            params: Network parameters.
            u, x, y, u2: Inputs to the network.
        Returns:
            Residual output.
        """
        Q_well = -5000  # m3/d, extraction by well
        k_field = 5  # Example constant for field
        s_xx = grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, y, u2)
        s_yy = grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, x, y, u2)
        res = k_field * (s_xx + s_yy) + self.well_function(u, x, y, Q_well)
        return res

    def well_function(self, u, x, y, Q_well):
        """
        Defines a function representing the well behavior in the residual network.
        Args:
            u, x, y: Input parameters.
            Q_well: Well output.
        Returns:
            The output of the well function.
        """
        return lax.cond(abs(x - u[0]) < 1.e-6, lambda _: lax.cond(abs(y - u[1]) < 1.e-6, lambda _: Q_well, lambda _: 0), lambda _: 0)


    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, res_batch):
        """
        Computes the residual loss.
        Args:
            params: Network parameters.
            res_batch: Batch of residual data.
        Returns:
            Residual loss value.
        """
        u_res, y_res = res_batch[0][:2]
        u2_res = res_batch[0][2] if len(res_batch[0]) == 3 else None
        s_r_pred = vmap(self.residual_net, (None, 0, 0, 0, 0))(params, u_res, y_res[:, 0], y_res[:, 1], u2_res)
        return np.mean(s_r_pred ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_operator_pi(self, params, res_batch, u2_res=None, bcs_batch=None):
        """
        Computes physics-informed operator loss.
        Args:
            params: Network parameters.
            res_batch: Batch of data for residual calculation.
            u2_res: Additional input, if any, for residual batch.
            bcs_batch: Batch of data for boundary conditions (unused in current implementation).
        Returns:
            Physics-informed loss value.
        """
        u_res, y_res = res_batch[0][:2]
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u_res, y_res[:, 0], y_res[:, 1])
        s_r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u_res, y_res[:, 0], y_res[:, 1])

        lam_bc, lam_r = 1.0, 0.01
        loss_bcs = np.mean((s_pred.flatten() - res_batch[1].flatten()) ** 2 * lam_bc)
        loss_res = np.mean(s_r_pred ** 2 * lam_r)
        return loss_bcs + loss_res

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        """
        Performs a single optimization step.
        Args:
            i: Current iteration number.
            opt_state: Current state of the optimizer.
            batch: Current batch of data.
        Returns:
            Updated optimizer state.
        """
        params = self.get_params(opt_state)
        loss_fn = self.loss_operator_pi if self.PI else self.loss_operator
        grad_fn = grad(loss_fn)
        g = grad_fn(params, batch)
        return self.opt_update(i, g, opt_state)

    def train(self, dataset_train, dataset_test, nIter=10000, earlystopping=False):
        """
        Trains the model on the provided datasets.
        Args:
            dataset_train: Training dataset.
            dataset_test: Testing dataset.
            nIter: Number of training iterations.
            earlystopping: Flag to enable early stopping.
        """
        data_iterator = iter(dataset_train)
        data_iterator_test = iter(dataset_test)
        best_val_loss, best_val_it = None, None
        pbar = trange(nIter)

        for it in pbar:
            batch = next(data_iterator)
            self.opt_state = self.step(next(self.itercount), self.opt_state, batch)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)
                loss_value = self.loss_operator(params, batch)
                loss_value_test = self.loss_operator(params, next(data_iterator_test))
                self.update_logs(loss_value, loss_value_test, it, pbar)

                if earlystopping and self.check_early_stopping(it, best_val_loss, best_val_it, loss_value_test):
                    break
             
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, u, y, u2 = None):
         if y.shape[1] == 2: 
                  xx, yy, xx2, yy2, xx3, yy3 = y[:,0],y[:,1], None, None, None, None
         elif y.shape[1] == 3: 
             xx, yy, xx2, yy2 , xx3, yy3= y[:,0],y[:,1], y[:,2],None, None, None
         elif y.shape[1] == 4: 
                 xx, yy, xx2, yy2 , xx3, yy3= y[:,0],y[:,1], y[:,2],y[:,3], None, None
         elif y.shape[1] == 6: 
                xx, yy, xx2, yy2 , xx3, yy3= y[:,0],y[:,1], y[:,2],y[:,3]  , y[:,4],y[:,5]      

         # Compute forward pass
         s_pred = vmap(self.operator_net, (None, 0, 0, 0, 0,0,0, 0,0 ))(params, u, xx, yy, xx2, yy2, xx3, yy3, u2)
         return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, u_res, y_res, u2_res = None):
         s_r_pred = vmap(self.residual_net, (None,  0, 0, 0, 0 ))(params, u_res, y_res[:,0], y_res[:,1], u2_res)
         return s_r_pred

    def count_params(self):
        params = self.get_params(self.opt_state)
        params_flat, _ = ravel_pytree(params)
        print("The number of model parameters is:",params_flat.shape[0])