# -*- coding: utf-8 -*-

"""
Test Script: E3_TimeDependentProblem

Purpose:
This test script is intended to assess the performance of the DeepONet-based emulator in handling time-dependent problems in groundwater flow. Specifically, it simulates the hydraulic head response in an aquifer subject to time-varying pumping rates from multiple wells. The test demonstrates the model's capability to handle dynamic input scenarios and predict the corresponding spatial and temporal changes in the hydraulic head.

Author: Maria Luisa Taccari (cnmlt@leeds.ac.uk)
"""

import jax
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
from jax.config import config
import pickle
# import jaxopt
from scipy.interpolate import interp1d

import matplotlib.patches as mpatches

training = True

# ----------------------------
# Data Loading Section
# ----------------------------


# Adjusted paths
base_path = "..\\data\\time_dependent\\2D_3wells\\"

u_train = np.load(base_path + "Y_train.npy")
u_test= np.load(base_path + "Y_test.npy")

u_train = u_train.reshape(u_train.shape[0], u_train.shape[1]*u_train.shape[2]*u_train.shape[3])
u_test = u_test.reshape(u_test.shape[0], u_test.shape[1]*u_test.shape[2]*u_test.shape[3])

v_train = np.load(base_path + "pumpingrate_train.npy")
v_test = np.load(base_path + "pumpingrate_test.npy")

v_train = v_train.reshape(v_train.shape[0], v_train.shape[1]*v_train.shape[2])
v_test = v_test.reshape(v_test.shape[0], v_test.shape[1]*v_test.shape[2])

# Given arrays
array1 = np.linspace(0, 1, 32)
array2 = np.linspace(0, 1, 32)
array3 = np.linspace(0, 1, 100)

# Create mesh grid
grid1, grid2, grid3 = np.meshgrid(array1, array2, array3, indexing='ij')

# Reshape and combine the grids
x_train = np.stack((grid1, grid2, grid3), axis=-1).reshape(-1,3)
x_test = x_train

# Display the shape of each dataset
print("Shape of u_train: ", u_train.shape)
print("Shape of v_train: ", v_train.shape)
print("Shape of u_test: ", u_test.shape)
print("Shape of v_test: ", v_test.shape)
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

#%%

Xmin = np.min(v_train)
Xmax = np.max(v_train)

dmin = np.min(u_train)
dmax = np.max(u_train)

u_train = 2.*(u_train - dmin)/(dmax - dmin) - 1.0
u_test = 2.*(u_test- dmin)/(dmax - dmin) - 1.0

v_train = 2.*(v_train - Xmin)/(Xmax - Xmin) - 1.0
v_test = 2.*(v_test- Xmin)/(Xmax - Xmin) - 1.0


#%%

def save_model(param, n):
    # Directory to save models
    dir_name = '../results/Time_2D_3wells'
    
    # If directory doesn't exist, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Saving the model
    filename = os.path.join(dir_name, 'model.' + str(n))
    with open(filename, 'wb') as file:
        pickle.dump(param, file)



def load_model(n):
    filename = './results/Time_2D_3wells/model.' + str(n)
    with open(filename, 'rb') as file:
        param = pickle.load(file)
    return param



from jax import jit, value_and_grad
from jax import random
from jax.example_libraries import optimizers


key = random.PRNGKey(1234)

initializer = jax.nn.initializers.glorot_normal()

def hyper_initial_WB(layers):
    L = len(layers)
    W = []
    b = []
    for l in range(1, L):
        in_dim = layers[l-1]
        out_dim = layers[l]
        std = np.sqrt(2.0/(in_dim+out_dim))
        weight = initializer(key, (in_dim, out_dim), jnp.float32)*std
        bias = initializer(key, (1, out_dim), jnp.float32)*std
        W.append(weight)
        b.append(bias)
    return W, b

# Initialize the parameters

def hyper_parameters_A(shape):
    return jnp.full(shape, 0.1, dtype=jnp.float32)

def hyper_parameters_amplitude(shape):
    return jnp.full(shape, 0.0, dtype=jnp.float32)

def hyper_parameters_freq1(shape):
    return jnp.full(shape, 0.1, dtype=jnp.float32)

def hyper_initial_frequencies(layers):

    L = len(layers)

    a = []
    c = []

    a1 = []
    F1 = []
    c1 = []
    
    for l in range(1, L):
        a.append(hyper_parameters_A([1]))
        c.append(hyper_parameters_A([1]))

        a1.append(hyper_parameters_amplitude([1]))
        F1.append(hyper_parameters_freq1([1]))
        c1.append(hyper_parameters_amplitude([1]))
    return a, c, a1, F1, c1#, a5, F5


def fnn_B(X, W, b, a, c, a1, F1, c1):
    inputs = X
    L = len(W)
    for i in range(L-1):
        outputs = jnp.dot(inputs, W[i]) + b[i]
        inputs = jnp.tanh(outputs)  # inputs to the next layer

    Y = jnp.dot(inputs, W[-1]) + b[-1]  
    return Y

def fnn_T(X, W, b, a, c, a1, F1, c1):  #, a3, F3, a4, F4, a5, F5
    inputs = X
    L = len(W)
    for i in range(L-1):
       # outputs = jnp.dot(inputs, W[i]) + b[i]
       # inputs = jnp.sin(outputs)  # inputs to the next layer     
       inputs =  jnp.cos(jnp.add(10*a[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c[i])) \
         + 10*a1[i]*jnp.sin(jnp.add(10*F1[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c1[i])) 
         # + 10*a2[i]*jnp.sin(jnp.add(20*F2[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c2[i]))  \
         #      + 10*a3[i]*jnp.sin(jnp.add(30*F3[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c3[i])) \
         
    Y = jnp.dot(inputs, W[-1]) + b[-1] 
    return Y

# #input dimension for Branch Net
u_dim = v_train.shape[1]

#output dimension for Branch and Trunk Net
G_dim = 50

#Branch Net
layers_f = [u_dim] + [50]*6 + [G_dim]

# Trunk dim
x_dim = 3

#Trunk Net
layers_x = [x_dim] + [50]*6 + [G_dim]

W_branch, b_branch = hyper_initial_WB(layers_f)
a_branch, c_branch, a1_branch, F1_branch , c1_branch = hyper_initial_frequencies(layers_f)
      
W_trunk,  b_trunk = hyper_initial_WB(layers_x)
a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk  = hyper_initial_frequencies(layers_x)
   

def predict(params, data):
    W_branch, b_branch, W_trunk, b_trunk = params
    v, x = data

    u_out_branch = fnn_B(v, W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch , c1_branch)
    u_out_trunk = fnn_T(x, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk )

    u_pred = jnp.einsum('ij,kj->ik',u_out_branch, u_out_trunk) # matmul
    return u_pred

def loss(params, data, u):
    W_branch, b_branch, W_trunk, b_trunk = params
    u_preds = predict(params, data)
    mse = jnp.mean((u_preds.flatten() - u.flatten())**2) 
    return mse

@jit
def update(params, data, u, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, data, u)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

# Defining an optimizer in Jax
num_epochs_adam = 10000000
num_epochs_tot = num_epochs_adam  + 2000

lr = 1e-3

opt_init, opt_update, get_params = optimizers.adam(lr)
opt_state = opt_init([W_branch, b_branch, W_trunk, b_trunk])

params = get_params(opt_state)
train_loss, test_loss = [], []
epo = []
start_time = time.time()
n = 0


if training == True:
        
    for epoch in range(num_epochs_adam):
        
        params, opt_state, loss_val = update(params, [v_train, x_train], u_train, opt_state)

        if epoch % 100 ==0:
          epoch_time = time.time() - start_time
          u_train_pred = predict(params, [v_train, x_train])
          err_train = jnp.mean(jnp.linalg.norm(u_train - u_train_pred, 2, axis=1)/\
                np.linalg.norm(u_train , 2, axis=1))
          u_test_pred = predict(params, [v_test, x_test])
          err_test = jnp.mean(jnp.linalg.norm(u_test - u_test_pred, 2, axis=1)/\
                np.linalg.norm(u_test , 2, axis=1))
          test_loss_val = loss(params, [v_test, x_test], u_test)
          test_loss.append(test_loss_val)
          train_loss.append(loss_val)
          epo.append(epoch)
          
          start_time = time.time()
          
          W_branch, b_branch, W_trunk, b_trunk = params
          
          
        if epoch % 100 == 0:
          print("Epoch {} | T: {:0.4f} | Train MSE: {:0.3e} | Test MSE: {:0.3e} ".format(
              epoch, epoch_time, loss_val, test_loss_val))
          
    params = get_params(opt_state)
    save_model(params,n)      
          
          
else:
    
    params = load_model(71)


pred = predict(params, [v_train, x_train])
u_test_pred = predict(params, [v_test, x_test])

# rescaling
pred = (pred+1.0)*(dmax-dmin)/2+dmin 
u_train = (u_train+1.0)*(dmax-dmin)/2+dmin

u_test_pred = (u_test_pred+1.0)*(dmax-dmin)/2+dmin
u_test = (u_test+1.0)*(dmax-dmin)/2+dmin

v_train = (v_train+1.0)*(Xmax-Xmin)/2+Xmin
v_test = (v_test+1.0)*(Xmax-Xmin)/2+Xmin


# reshape
u_train = u_train.reshape(1000, 32, 32, 100)
u_test = u_test.reshape(200, 32, 32, 100)

pred = pred.reshape(1000, 32, 32, 100)
u_test_pred = u_test_pred.reshape(200, 32, 32, 100)


#%% PLOTS

idx_1_test = 20
idx_2_test = 60
idx_3_test = 100
idx_4_test = 180

# Plot the solutions

fig, ax = plt.subplots(3,4)
fig.suptitle('Dataset vs. Prediction', fontsize=16)
fig.set_figwidth(17)
fig.set_figheight(10)

ax[0, 0].set_ylabel('Test data', fontsize=12)

im1 = ax[0,0].imshow(u_test[idx_3_test,:,:,10])
im2 = ax[0,1].imshow(u_test[idx_3_test,:,:,30])
im3 = ax[0,2].imshow(u_test[idx_3_test,:,:,60])
im4 = ax[0,3].imshow(u_test[idx_3_test,:,:,80])

ax[1, 0].set_ylabel('Test prediction', fontsize=12)

im5 = ax[1,0].imshow(u_test_pred[idx_3_test,:,:,10])
im6 = ax[1,1].imshow(u_test_pred[idx_3_test,:,:,30])
im7 = ax[1,2].imshow(u_test_pred[idx_3_test,:,:,60])
im8 = ax[1,3].imshow(u_test_pred[idx_3_test,:,:,80])

ax[2, 0].set_ylabel('Relative error', fontsize=12)

im9  = ax[2,0].imshow(np.abs(u_test[idx_3_test,:,:,10] - u_test_pred[idx_3_test,:,:,10]))
im10 = ax[2,1].imshow(np.abs(u_test[idx_3_test,:,:,30] - u_test_pred[idx_3_test,:,:,30]))
im11 = ax[2,2].imshow(np.abs(u_test[idx_3_test,:,:,60] - u_test_pred[idx_3_test,:,:,60]))
im12 = ax[2,3].imshow(np.abs(u_test[idx_3_test,:,:,80] - u_test_pred[idx_3_test,:,:,80]))

for idx, col in enumerate([10, 30, 60, 80]):

    frame = np.abs(u_test[idx_3_test,:,:,col] - u_test_pred[idx_3_test,:,:,col])

    im = ax[2,idx].imshow(frame)
    # fig.colorbar(im, ax=ax[2, idx], fraction=0.046, pad=0.04)
    
    cbar = fig.colorbar(im, ax=ax[2, idx], orientation='vertical', pad=0.1)
    cbar.ax.tick_params(labelsize=8)

cax1 = fig.add_axes([0.92, 0.69, 0.01, 0.23])  # Position of colorbar for ax[0,0]
cax2 = fig.add_axes([0.92, 0.38, 0.01, 0.23])  # Position of colorbar for ax[1,0]

fig.colorbar(im1, cax=cax1)
fig.colorbar(im5, cax=cax2)

ax[0, 0].set_title('time = 10')
ax[0, 1].set_title('time = 30')
ax[0, 2].set_title('time = 60')
ax[0, 3].set_title('time = 80')

fig_name = "train_test_all_space.png"
plt.savefig(fig_name, dpi=300)


#%%

fig, ax = plt.subplots(3,4)
fig.suptitle('Dataset vs. Prediction', fontsize=16)
fig.set_figwidth(17)
fig.set_figheight(10)

ax[0, 0].set_ylabel('Test data', fontsize=12)

# Plot test data
levels = [-2.5, -2 , -1.4, -1.2, -1, -0.8 , -0.5,]
for idx, col in enumerate([10, 30, 60, 80]):
    frame = u_test[idx_3_test,:,:,col]
    im = ax[0,idx].imshow(frame, cmap='afmhot')
    cset = ax[0,idx].contour(frame, cmap='Set1_r', linewidths=2, levels=levels)
    ax[0, idx].set_title(f"time = {col}")

 
# Create a legend for the contour levels
patches = [mpatches.Patch(color=plt.cm.Set1_r(i / (len(levels)-1)), label=f"{level:.2f}") for i, level in enumerate(levels)]
ax[0, -1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
ax[1, 0].set_ylabel('Test prediction', fontsize=12)

# Plot test predictions
for idx, col in enumerate([10, 30, 60, 80]):
    frame = u_test_pred[idx_3_test,:,:,col]
    im = ax[1,idx].imshow(frame, cmap='afmhot')
    ax[1,idx].contour(frame, cmap='Set1_r', linewidths=2, levels=levels)

# Create a legend for the contour levels
patches = [mpatches.Patch(color=plt.cm.Set1_r(i / (len(levels)-1)), label=f"{level:.2f}") for i, level in enumerate(levels)]
ax[1,-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
ax[2, 0].set_ylabel('Absolute error', fontsize=12)

# Relative error
for idx, col in enumerate([10, 30, 60, 80]):
    frame = np.abs(u_test[idx_3_test,:,:,col] - u_test_pred[idx_3_test,:,:,col])
    im = ax[2,idx].imshow(frame)    
    cbar = fig.colorbar(im, ax=ax[2, idx], orientation='horizontal', pad=0.1)
    cbar.ax.tick_params(labelsize=8)



fig_name = "train_test_all_space_contour.png"
plt.savefig(fig_name, dpi=300)
plt.show()

#%%

fig, ax = plt.subplots(2,4)
fig.suptitle('Dataset vs. Prediction', fontsize=16)
fig.set_figwidth(17)
fig.set_figheight(10)

im1 = ax[0,0].plot(u_test[idx_1_test,15,5,:],label='Data')
im2 = ax[0,1].plot(u_test[idx_2_test,15,5,:])
im3 = ax[0,2].plot(u_test[idx_3_test,15,5,:])
im4 = ax[0,3].plot(u_test[idx_4_test,15,5,:])

im5 = ax[0,0].plot(u_test_pred[idx_1_test,15,5,:],'--',label='Prediction')
im6 = ax[0,1].plot(u_test_pred[idx_2_test,15,5,:],'--')
im7 = ax[0,2].plot(u_test_pred[idx_3_test,15,5,:],'--')
im8 = ax[0,3].plot(u_test_pred[idx_4_test,15,5,:],'--')

ax[0,0].legend()


ax[1, 0].set_ylabel('Absolute error', fontsize=14)

im9 = ax[1,0].plot(np.abs(u_test[idx_1_test,15,5,:] - u_test_pred[idx_1_test,15,5,:]),'.--')
im10 = ax[1,1].plot(np.abs(u_test[idx_2_test,15,5,:] - u_test_pred[idx_2_test,15,5,:]),'.--')
im11 = ax[1,2].plot(np.abs(u_test[idx_3_test,15,5,:] - u_test_pred[idx_3_test,15,5,:]),'.--')
im14 = ax[1,3].plot(np.abs(u_test[idx_4_test,15,5,:] - u_test_pred[idx_4_test,15,5,:]),'.--')

# Set y-axis log scale for the third row
ax[1, 0].set_yscale('log')
ax[1, 1].set_yscale('log')
ax[1, 2].set_yscale('log')
ax[1, 3].set_yscale('log')

fig_name = "train_test_all_time.png"
plt.savefig(fig_name, dpi=300)


#%% Errors train

# Compute L1 error
l1_error = np.mean(np.abs(u_train - pred), axis=1)
# Compute L2 error
l2_error = np.sqrt(np.mean((u_train - pred)**2, axis=1))
# Compute L infinity error
l_infinit_error = np.max(np.abs(u_train - pred) , axis=1)
# Compute relative error
relative_error = np.mean(np.abs(u_train - pred) /np.abs(u_train) , axis=1)
relative_error_test = np.mean(np.abs(u_test - u_test_pred) /np.abs(u_test) , axis=1)


fig1 ,ax1  = plt.subplots(1,1)

ax1.plot(train_loss,'-b',label='Train')
ax1.plot(test_loss,'-r',label='Test')
ax1.legend()
ax1.set_yscale('log')

fig_name = "loss.png"
plt.savefig(fig_name, dpi=300)


