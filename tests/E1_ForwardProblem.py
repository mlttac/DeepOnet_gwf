# -*- coding: utf-8 -*-
"""
Test Script: E1_ForwardProblem

Purpose:
This script is designed to test the DeepONet-based emulator's ability to predict the distribution of hydraulic head in a heterogeneous confined aquifer. The test focuses on a scenario with a fully penetrating well that starts pumping at a constant rate. The aim is to evaluate the model's accuracy and efficiency in simulating groundwater flow under these specific conditions.

Author: Maria Luisa Taccari (cnmlt@leeds.ac.uk)
"""

import sys
sys.path.append('C:\codes\DeepONet')

import os
import numpy as np
import jax.numpy as jnp
import timeit
import matplotlib.pyplot as plt

from model.preprocessing import (sampling, normalization, cart2pol, create_extra_inputs, normalization_inversed)
from model.deeponet import DeepONet, DataGenerator
from model.prediction_error import error_full_resolution, predict_function, error_comparison
from model.utilis import (plotfigs, plot_sampling, reshape_sy, plot_loss, prediction_well_time, save_model, load_model)
from model.plot_errorbars import errorbars
from model.feature_expansion import HarmonicFeatureExpansionY

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def run_model_parameters():
    global input_y, input_u, samplingT, TRAINING_ITERATIONS, arch, weights
    global id_run, howmany_train, training_batch_size, Pxy, nlayers_trunk
    global nw_trunk, act_trunk, nw_last, kernel_size_branch, nlayers_fnn_branch
    global nw_fnn_branch, cnn_branch, learning_rate, normalize, SamplingPoint
    global sampling_strategy, k_field, filename_data

    # Set model parameters
    input_y = 'cartesian_coord'
    input_u = 'k'
    samplingT = ''
    TRAINING_ITERATIONS = 100000
    arch = 'CNN'
    weights = 'no_weights'
    id_run = 0
    howmany_train = 1000
    training_batch_size = 100
    Pxy = 128
    nlayers_trunk = 5
    nw_trunk = 150
    act_trunk = 'relu'
    nw_last = 150
    kernel_size_branch = 5
    nlayers_fnn_branch = 2
    nw_fnn_branch = 150
    cnn_branch = [16, 16, 16, 16, 64]
    learning_rate = 5e-4
    normalize = True
    SamplingPoint = False
    sampling_strategy = True
    k_field = 'Random field'
    filename_data = 'Forward1'


def load_dataset(filename_data):
    try:
        return np.load("../data/" + filename_data + ".npz")
    except FileNotFoundError:
        return np.load("/data/" + filename_data + ".npz")

def data_preprocessing(dataset):
    # Data preprocessing functions are added here
    global U_train, Y_train, S_train, U_test, Y_test, S_test, num_train, num_test, U_scaler_test
    global Nx, Ny, Nt, P_test, du, dy, h, m, du_fnn, cnn_dim, trunk_layers, branch_layers
    global y_train, y_test, s_train, s_test
    
    # Training data extraction with shapes
    howmany_train = 1000
    U_train = dataset["U_train"][:howmany_train, :, :]
    Y_train = dataset["Y_train"][:howmany_train, :, :]
    S_train = dataset["s_train"][:howmany_train, :, :]

    # Number of test samples
    howmany_test = 200
    U_test = dataset["U_test"][:howmany_test, :, :]
    Y_test = dataset["Y_test"][:howmany_test, :, :]
    S_test = dataset["s_test"][:howmany_test, :, :]

    # Other variables from the provided code
    num_train = U_train.shape[0]
    num_test = U_test.shape[0]
    h = int(np.sqrt(U_train.shape[1])) if len(U_train.shape) == 3 else U_train.shape[1]
    Nx = h
    Ny = h
    ds = 1
    Nt = 1
    m = h * h
    P_test = Nx * Ny * Nt
    

    du = U_train.shape[-1]
    dy = Y_train.shape[-1]
    
    # Normalize the data
    _, U_train = normalization(U_train)
    U_scaler_test, U_test = normalization(U_test)
    _, Y_train = normalization(Y_train)
    _, Y_test = normalization(Y_test)
    S_train = np.expand_dims(S_train, axis = -1)
    S_test = np.expand_dims(S_test, axis = -1)
    norm_log = True if normalize == 'Log_target' else False
    _, S_train = normalization(S_train, log = norm_log)
    _, S_test = normalization(S_test, log = norm_log)


    # Point sampling
    
    # find divosors of total number of points to pick up P
    Nxyt = h*h*Nt
    divisors = []
    for i in range(1, Nxyt//2 + 1):
        if Nxyt % i == 0:
            divisors.append(i)
    divisors.append(Nxyt)
    # print("The possible choices for P are {}".format(divisors))
    P = Pxy*Nt if samplingT == 'all' else Pxy
    s_train = np.zeros((num_train,P,ds))
    y_train = np.zeros((num_train,P,dy))
    # only sampling training data
   

    _, _, well_coord_train_nw, _, _ = create_extra_inputs(U_train, num_train, h, Nx=Nx, Ny=Ny, Nt=Nt,  channelWell = 1 )
    _, _, well_coord_test_nw, _, _  = create_extra_inputs(U_test, num_test, h, Nx=Nx, Ny=Ny, Nt=Nt,  channelWell = 1 )

    U_train = U_train[:,:,[0]]
    U_test = U_test[:,:,[0]]
    du = U_train.shape[-1]
    dy = Y_train.shape[-1]
    
    sampling_box_width = 5  
    for i in range(0,num_train):
       _, _, s_train[i,:,:], y_train[i,:,:] = sampling(S_train[i,:], Y_train[i], U_train[i], Nx=Nx, Ny=Ny, Nt=Nt, box_width = sampling_box_width,sampling_strategy = sampling_strategy, P=Pxy, ds=ds, dy=dy, samplingT = samplingT,
                                                       x_well = int(well_coord_train_nw[i,0,0]), y_well =int(well_coord_train_nw[i,0,1]) )

    s_test, y_test = reshape_sy(S_test,Y_test,num_test,P_test,ds, dy)
    
    
    U_train = jnp.asarray(U_train).reshape(num_train, m * du)
    U_test = jnp.asarray(U_test).reshape(num_test, m * du)
    s_train = jnp.asarray(s_train).reshape(num_train*P,ds)
    y_train = jnp.asarray(y_train).reshape(num_train*P,dy)
    s_test = jnp.asarray(s_test).reshape(num_test*P_test,ds)
    y_test = jnp.asarray(y_test).reshape(num_test*P_test,dy)
      
    return

def create_folder_for_results():
    global path_results_id
    
    path_results = "../results/Forward1"
    if not os.path.exists(path_results):
        os.mkdir(path_results)
        
    path_results_id = os.path.join(path_results, str(id_run))
    if not os.path.exists(path_results_id):
        os.mkdir(path_results_id)
    return path_results_id

def train_model():
    global model 
    
    u2_train = None
    u2_test = None
    train_dataset = DataGenerator( u = U_train, y = y_train, s = s_train, u2 = u2_train, batch_size= training_batch_size )
    test_dataset = DataGenerator( u = U_test, y = y_test, s= s_test, u2 = u2_test, batch_size=  training_batch_size)
    train_dataset = iter(train_dataset)
    test_dataset = iter(test_dataset)
    du_fnn = U_train.shape[-1]
    branch_layers = np.concatenate(([du_fnn], np.repeat(nw_fnn_branch, nlayers_fnn_branch), [nw_last ]))   
    trunk_layers = np.concatenate(([y_train.shape[-1]], np.repeat(nw_trunk, nlayers_trunk), [nw_last ]))   
    PI = False
    weights = 'no_weights'
    normalized_loss = True if normalize == False or normalize == 'Only_input' else False
    decoder = 'multiplication'
    # Initialize model
    model = PI_DeepONet(arch, weights, branch_layers, trunk_layers, du, PI, learning_rate,
                        kernel_size_branch, cnn_branch, act_trunk, decoder, normalized_loss)

    start_time = timeit.default_timer()
    model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS) #bcs_dataset, res_dataset
    elapsed = timeit.default_timer() - start_time
    print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

    plot_loss(model.loss_log, loss2 = model.loss_log_test, loss2_label = 'Test loss',  path = path_results_id) 
    params = model.get_params(model.opt_state)

    save_model(params,path_results_id, id_run)      

def predict_model(params, U_test, y_test, num_test, Nx, Ny, Nt):
    uCNN_super_all_test = np.zeros_like(S_test).reshape(num_test, Nx,Ny,Nt)
    start_time = timeit.default_timer()
    for idx in range(0, num_test):
        index = np.arange(idx * P_test,(idx + 1) * P_test)
        u2_test_index = None
        u_test = jnp.tile(U_test[idx,:], P_test).reshape(P_test, -1)
        uCNN_super_all_test[idx,:,:,:]  = model.predict_s(params,  u_test, y_test[index,:], u2_test_index)[:,None].reshape(Nx,Ny,Nt)
    
    print('Forward time: %.6f secs' % ((timeit.default_timer() - start_time)/num_test))
    
    
    
    l2_error_test, percent_error_test = error_full_resolution(uCNN_super_all_test,  S_test,  tag='test', num_train=num_test,Nx=Nx, Ny=Ny,Nt=Nt)
    
    # use the whole ground truth image for prediction
    P_all = h* h 
    Y_train_ = Y_train.reshape(howmany_train*P_all,dy)
    uCNN_super_all_train = np.zeros_like(S_train).reshape(num_train, Nx,Ny,Nt)
    for idx in range(0, num_train):
        index = np.arange(idx * P_all,(idx + 1) * P_all)
        u2_train_index = None
        u_train = jnp.tile(U_train[idx,:], P_all).reshape(P_all, -1)
        uCNN_super_all_train[idx,:,:,:]  = model.predict_s(params,  u_train, Y_train_[index,:], u2_train_index)[:,None].reshape(Nx,Ny,Nt)
    
    l2_error_train, percent_error_train = error_full_resolution(uCNN_super_all_train,  S_train,  tag='train', num_train=num_train,Nx=Nx, Ny=Ny,Nt=Nt)
    
    
    errors =(
             np.mean(l2_error_train), 
             np.std(l2_error_train) , 
             np.mean(l2_error_test), 
             np.std(l2_error_test),
             np.mean(percent_error_train), 
             np.std(percent_error_train) , 
             np.mean(percent_error_test), 
             np.std(percent_error_test) 
             )

    relative_l2_error = error_comparison(uCNN_super_all_test,  S_test,  tag='test', num_test=num_test,Nx=Nx, Ny=Ny,Nt=Nt)
    
    print("The average test error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(relative_l2_error),
                                                                                                                         np.std(relative_l2_error),
                                                                                                        np.min(relative_l2_error), 
                                                                                                        np.max(relative_l2_error)))
                                   
                               
    return uCNN_super_all_test

def visualize_predictions(uCNN_super_all_test):
    U_test_orig = normalization_inversed(U_scaler_test , U_test.reshape(-1,32,32,2))
    
    for i in range(10):
         idx = i
         index = np.arange(idx * P_test, (idx + 1) * P_test)
         u_test = jnp.tile(U_test[idx, :], P_test).reshape(P_test, -1)
         s_pred = model.predict_s(params,  u_test, y_test[index, :], None)[:, None].reshape(Nx, Ny, Nt)
     
         try:
             kplot = U_test_orig[idx, :].reshape(h, h)
         except:
             kplot = U_test_orig.reshape(num_test, m, du)[idx, :, 0].reshape(h, h)
     
         # Creating well location map from the reshaped test data
   
         # Getting a horizontal cross section along one coordinate of the well
         y_well_coord = 16  # Choosing the first well's y coordinate
     
         fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 6), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
     
         # Input Hydraulic conductivity
         im1 = axs[0].imshow(kplot, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
         axs[0].scatter(16 / h, 16 / h, color='red')  # well locations overlay
         axs[0].set_title('Input ($m^3/day$)', fontsize=16)
         cb1 = fig.colorbar(im1, ax=axs[0], orientation='horizontal', pad=0.1, aspect=30)
         cb1.formatter.set_powerlimits((0, 0))
         cb1.update_ticks()
     
         vmin, vmax = np.min(s_test[index, :].reshape(h, h)), np.max(s_test[index, :].reshape(h, h))
     
         # Ground Truth
         im2 = axs[1].imshow(s_test[index, :].reshape(h, h), cmap='afmhot_r', origin='lower', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
         axs[1].contour(s_test[index, :].reshape(h, h), cmap='Set1_r', linewidths=2, extent=[0, 1, 0, 1])  # Contour lines
         axs[1].set_title('Ground Truth (m)', fontsize=16)
         fig.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.1, aspect=30)
     
         # Prediction
         im3 = axs[2].imshow(s_pred.reshape(h, h), cmap='afmhot_r', origin='lower', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
         axs[2].contour(s_pred.reshape(h, h), cmap='Set1_r', linewidths=2, extent=[0, 1, 0, 1])  # Contour lines
         axs[2].set_title('Prediction (m)', fontsize=16)
         fig.colorbar(im3, ax=axs[2], orientation='horizontal', pad=0.1, aspect=30)
     
         # Absolute error
         im4 = axs[3].imshow(np.abs(s_pred.reshape(h, h) - s_test[index, :].reshape(h, h)), cmap='RdGy_r', origin='lower', extent=[0, 1, 0, 1])
         axs[3].set_title('Absolute Error (m)', fontsize=16)
         fig.colorbar(im4, ax=axs[3], orientation='horizontal', pad=0.1, aspect=30)
     
         # Cross-section along well
         axs[4].plot(s_pred.reshape(h, h)[y_well_coord, :], color="orange")  # Prediction
         axs[4].plot(s_test[index, :].reshape(h, h)[y_well_coord, :], color="green")  # Ground truth
         axs[4].legend(["Prediction", "Ground Truth"])
         axs[4].set_xlabel('x coordinate')
         axs[4].set_ylabel("Head")
         axs[4].set_title('Cross-section along well', fontsize=16)
     
    
         plt.savefig(os.path.join(path_results_id, str(i)), dpi=300)
         plt.close(fig) 

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(30, 12), gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
    
    for i in range(5):
         idx = i
         index = np.arange(idx * P_test, (idx + 1) * P_test)
         u_test = jnp.tile(U_test[idx, :], P_test).reshape(P_test, -1)
         s_pred = model.predict_s(params,  u_test, y_test[index, :], None)[:, None].reshape(Nx, Ny, Nt)
     
         try:
             kplot = U_test_orig[idx, :].reshape(h, h)
         except:
             kplot = U_test_orig.reshape(num_test, m, du)[idx, :, 0].reshape(h, h)
     
         # Creating well location map from the reshaped test data
         well_map = np.where(U_test.reshape(num_test, m, du)[idx, :, 1].reshape(h, h) != 0)
     
         # Input Hydraulic conductivity
         im1 = axs[0, i].imshow(kplot, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
         axs[0, i].scatter(16/h, 16/h, color='red')  # well locations overlay
         axs[0, i].set_title('Input ($m^3/day$)', fontsize=16)
         cb1 = fig.colorbar(im1, ax=axs[0, i], orientation='horizontal', pad=0.1, aspect=30)
         cb1.formatter.set_powerlimits((0, 0))
         cb1.update_ticks()
     
         vmin, vmax = np.min(s_test[index, :].reshape(h, h)), np.max(s_test[index, :].reshape(h, h))
     
         # Ground Truth
     
         im2 = axs[1, i].imshow(s_test[index, :].reshape(h, h), cmap='afmhot_r', origin='lower', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
         axs[1, i].contour(s_test[index, :].reshape(h, h), cmap='Set1_r', linewidths=2, extent=[0, 1, 0, 1])  # Contour lines
         axs[1, i].set_title('Ground Truth (m)', fontsize=16)
         fig.colorbar(im2, ax=axs[1, i], orientation='horizontal', pad=0.1, aspect=30)
         
    plt.savefig(os.path.join(path_results_id, 'allInputs'), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_model_parameters()
    dataset = load_dataset(filename_data)
    data_preprocessing(dataset)
    path_results = create_folder_for_results()
    train_model()
    params = load_model(path_results, id_run)
    uCNN_super_all_test = predict_model(params, U_test, Y_test, num_test, Nx, Ny, Nt)
    visualize_predictions(uCNN_super_all_test)
