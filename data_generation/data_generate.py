# -*- coding: utf-8 -*-
"""
Created on Wed Apr 6 10:37:29 2022

Script Description:
This script is dedicated to generating datasets for training the Deep Operator Network (DeepONet).
It leverages 'fdm3t', a time-dependent 3D finite difference model implemented in Python,
for simulating transient groundwater flow in a homogeneous confined aquifer influenced by a fully penetrating well starting to pump at a constant rate from t=0.

For more details on the fdm3t function and its implementation, please refer to:
https://github.com/Olsthoorn/FDModelBuildingCourseGroundwaterFlow

Author: Maria Luisa Taccari (cnmlt@leeds.ac.uk)
"""

import os
import sys
import numpy as np
import matplotlib.pylab as plt
import mfgrid
from fdm_t import fdm3t
from fdm import fdm3
import math
import random_field as rf

# Set the working directory to the 'data_generation' folder within the 'DeepONet' directory.
os.chdir(r'C:/codes/DeepONet/data_generation')

# Ensure custom modules are available in the path.
myModules = 'FDsolver'
if myModules not in sys.path:
    sys.path.insert(0, myModules)



#%% Main Function

def well_loc(Nx, Ny, pos):
    """
    Determines the location of a well based on specified strategy.

    Parameters:
    - Nx (int): Number of grid cells in the x-direction.
    - Ny (int): Number of grid cells in the y-direction.
    - pos (int): Strategy for positioning the well.
                 0 = Random position, excluding boundaries.
                 1 = Center of the grid.
                 2 = Middle of the grid, y at a quarter of the domain.
                 3 = Middle of the grid, y at three quarters of the domain.
                 4 = Middle of the grid, x at a quarter of the domain.
                 >4 = Divide the domain into 'pos' number of steps and place the well randomly among these positions.

    Returns:
    - wloc (ndarray): A 2D array with a flag indicating the well's position.
    - well_x (int): x-coordinate of the well.
    - well_y (int): y-coordinate of the well.
    """

    if pos == 0: 
        # Random position, but not on the boundaries
        well_x = np.random.randint(1, Nx-1)
        well_y = np.random.randint(1, Ny-1)
    elif pos == 1:
        # Central position
        well_x = Nx // 2
        well_y = Ny // 2
    elif pos == 2:
        # Center horizontally, a quarter down vertically
        well_x = Nx // 2
        well_y = Ny // 4
    elif pos == 3:
        # Center horizontally, three quarters down vertically
        well_x = Nx // 2
        well_y = 3 * Ny // 4
    elif pos == 4:
        # Center vertically, a quarter from the left horizontally
        well_x = Ny // 4
        well_y = Nx // 2
    else:
        # Divide the domain into equal parts and choose a random position
        nsteps = pos
        x_pos = np.linspace(start=Nx / (nsteps + 1), stop=Nx, num=nsteps, endpoint=False, dtype=np.int16)
        y_pos = np.linspace(start=Ny / (nsteps + 1), stop=Ny, num=nsteps, endpoint=False, dtype=np.int16)
        well_x = np.random.choice(x_pos)
        well_y = np.random.choice(y_pos)

    # Flag the well's position on the grid
    wloc = np.zeros((Nx, Ny))
    wloc[well_y, well_x] = 1
    return wloc, well_x, well_y


def well_time(s=32, pos=9, Nt=32, k_type='uniform', n_classes=[1, 5], infiltration_type='well', nwells=1, inputK=None):
    """
    Simulates the effect of well operations over time in a confined aquifer, taking into account various
    hydraulic conductivity fields and well infiltration strategies. This function is pivotal for generating
    the training datasets for the DeepONet model.

    Parameters:
    - s (int): Grid size in both x and y directions.
    - pos (int): Strategy for well positioning (0 for random, 1 for center, etc.).
    - Nt (int): Number of time steps for simulation.
    - k_type (str): Type of hydraulic conductivity field ('uniform', 'interface', etc.).
    - n_classes (list): Range of classes for discretizing the conductivity field.
    - infiltration_type (str): Type of infiltration source ('rain' or 'well').
    - nwells (int): Number of wells to simulate.
    - inputK (ndarray): Optional pre-defined hydraulic conductivity field.

    Returns:
    - U (ndarray): Array containing the input features for the model.
    - phi_store (ndarray): Array containing the simulated hydraulic heads at each time step.
    - t_store (list): List of time steps at which the hydraulic heads were simulated.
    """

    # Aquifer properties and simulation parameters
    D = 50  # Aquifer thickness in meters
    S = 0.01  # Elastic storage coefficient of the aquifer
    ss = S / D  # Specific elastic storage coefficient (1/m)
    Q = -5000  # Extraction rate in m3/day (negative for extraction)
    R = 8000  # Extent of the model in meters

    # Grid dimensions
    Nx, Ny = s, s
    xmin, xmax = 0, R
    ymin, ymax = 0, R
    t_stop = 200  # Simulation time
    
    # Hydraulic conductivity values
    k1 = 5  # First hydraulic conductivity value (m/day)
    k2 = 25  # Second hydraulic conductivity value (m/day)
            
    # Generate the hydraulic conductivity field based on the selected type
    K = inputK if inputK is not None else None
    if K is None:
        if k_type == 'uniform':
            K = np.full((s, s), np.random.uniform(k1, k2))
        elif k_type == 'interface':
            # Assume k_discrete2 function generates interface type conductivity
            _, _, K = rf.k_discrete2(Nx, Ny, np.random.uniform(k1, k2), np.random.uniform(k1, k2))
        elif k_type == 'classes':
            # Assume create_Krf function generates class-based conductivity
            n_cl = np.random.randint(n_classes[0], n_classes[1] + 1)
            K = rf.create_Krf(discrete=True, minK=k1, maxK=k2, n_classes=n_cl, size=s)
        elif k_type == 'rf':
               n_cl = np.random.randint( n_classes[0], n_classes[1] + 1 ) #+1 to be inclusive of n_classes
               K = rf.create_Krf(alpha=-4.0, discrete = False, minK = k1 , maxK = k2, n_classes = n_cl, size = s,  reshuffle_k = False)
            
        elif k_type == 'chunks':
               K = rf.split_chunks()
        elif k_type == 'channels': 
               k2 = 500
               K = rf.generate_channelized_array(size=32, k1=k1, k2=k2)

    # Create a grid for the model
    gr = mfgrid.Grid(np.linspace(xmin, xmax, Nx + 1), np.linspace(ymin, ymax, Ny + 1), np.array([0, -D]), axial=False)

    # Set specific storage for the entire grid
    Ss = gr.const(ss)
    
    # Initialize flow and head boundary conditions
    FQ = gr.const(0)
    FH = gr.const(0)
    IBOUND = gr.const(1)  # No fixed heads, all cells are active
    
    # Adjust FQ based on the type of infiltration
    if infiltration_type == 'rain':
        FQ = gr.const(-Q/(s*s))
    elif infiltration_type == 'well':
        well_locs = np.zeros(shape=(s, s))
        nwells = np.random.randint(1, nwells+1)
        for _ in range(nwells):
            wloc, well_x, well_y = well_loc(Nx, Ny, pos)
            FQ[:, well_y, well_x] = -Q
            well_locs += wloc


    # Simulate the groundwater flow with the finite difference model
    if Nt is not None:
        # Transient simulation
        t_whole = np.linspace(start=0, stop=t_stop, num=Nt + 1, endpoint=True)
        Out = fdm3t(gr, t_whole, (K, K, K), Ss, FQ, FH, IBOUND, epsilon=1)
        t_store = t_whole[1:]  # Exclude initial state at t=0
        phi_store = Out.Phi[1:, :, :, :].reshape(len(t_store), Ny, Nx)
    else:
        # Steady state simulation
        Out_steady = fdm3(gr, (K, K, K), FQ, FH, IBOUND)
        t_store = [1]
        phi_store = Out_steady.Phi

    return (K, well_locs), phi_store, t_store


#%% Dataset creation

if __name__ == "__main__":
       
       # DEFINE THE INPUTS:
       outputName = "../data/Forward1_channelized_k2_500.npz"
       num_train = 1000
       num_test = 200
       s = 32
       Nt = 1
       dy = 2 if Nt == 1 else 3 #(x,y,t)
       dy=2
       k_type = 'channels'
       infiltration_type= 'well'
       nwells= 1
              
       du = 2 
       du2 = 4
       realisations = [[num_train, "Train"], [num_test, "Test"]]      
       U_train = []
       U_train2 = []
       Y_train = []
       s_train = []
       U_test = []
       U_test2 = []
       Y_test = []
       s_test = []
       folders = [[U_train,  Y_train, s_train], [U_test, Y_test, s_test]]
       count = 0
       
       for realisation, folder in zip(realisations, folders):

              for iter in range(realisation[0]):
                     print("\n\nRunning Iter: " + str(realisation) + "\n=============================================================================")

                     if realisation[1] == "Train":
                            pos_w = 0
                            n_classes =  [2, 5]
                     else:
                            pos_w = 0
                            n_classes = [2, 5]
                     count +=1
                     
                     U, S, tsteps = well_time(s, pos_w, Nt, k_type ,n_classes , infiltration_type,  nwells )
                     folder[0].append(U)      
                     S = np.einsum('kij->ijk', S) 
                     folder[2].append(S.reshape(s,s,Nt))
                    
              x_ = np.linspace(0., 1., s)
              y_ = np.linspace(0., 1., s)

       if Nt!= 1:
              tsteps = np.linspace(0., 1., 5)
              XX, YY, TT = np.meshgrid(x_, y_, tsteps, indexing='ij')
              y_stacked = np.hstack((XX.flatten()[:,None], YY.flatten()[:,None], TT.flatten()[:,None]))
              y_stacked = y_stacked.reshape(s*s,len(tsteps),3)
              Y_train   = np.repeat(y_stacked[np.newaxis, :, :, :], num_train, axis=0)
              Y_test   = np.repeat(y_stacked[np.newaxis, :, :], num_test, axis=0)
       else: 
              x_ = np.linspace(0., 1., s)
              y_ = np.linspace(0., 1., s)
              XX, YY = np.meshgrid(x_, y_)
              y_stacked = np.hstack((XX.flatten()[:,None], YY.flatten()[:,None]))
              Y_train   = np.repeat(y_stacked[np.newaxis, :, :], num_train, axis=0)
              Y_test   = np.repeat(y_stacked[np.newaxis, :, :], num_test, axis=0)


                   
       np.savez_compressed(outputName, U_train=np.array(U_train).reshape(num_train, s*s, du), 
                        Y_train=np.array(Y_train).reshape(num_train, s*s*Nt, dy),
                        s_train=np.array(s_train),
                        U_test=np.array(U_test).reshape(num_test, s*s, du), 
                        Y_test=np.array(Y_test).reshape(num_test, s*s*Nt, dy),
                        s_test=np.array(s_test))
    
    
