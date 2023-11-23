# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:20:56 2022
Generates various types of hydraulic conductivity fields for groundwater modeling simulations.
Reference for random fields: https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/

Author: Maria Luisa Taccari (cnmlt@leeds.ac.uk)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from scipy.ndimage import gaussian_filter


# RBF Function: Radial Basis Function for generating Gaussian Process samples
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)

# k_discrete2 Function: Generates a discontinuity in a domain with 2 classes

def k_discrete2(Nx, Ny, k1, k2, length_scale=0.2):
    """
    Generates a hydraulic conductivity field with a discontinuity, dividing the domain into two classes.

    Parameters:
    - Nx, Ny (int): Dimensions of the grid.
    - k1, k2 (float): Hydraulic conductivity values for the two classes.
    - length_scale (float): Length scale parameter for the Gaussian process.

    Returns:
    - x (ndarray): X-coordinates of the grid.
    - ffn_norm (ndarray): Normalized function values determining the class boundary.
    - k (ndarray): Generated hydraulic conductivity field.
    """

    # Number of points for Gaussian Process (GP) sampling
    N = 512
    xmin, xmax = 0, 1

    # Set up GP parameters and generate a covariance matrix
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))

    # Generate a GP sample and create an interpolation function
    gp_sample = np.dot(L, np.random.normal(size=(N,)))
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Normalize the function values for grid dimensions
    y = np.linspace(xmin, xmax, Ny)
    ffn = f_fn(y)
    ffn_norm = (32 * (ffn - np.min(ffn)) / np.ptp(ffn)).astype(int)

    # Initialize the conductivity field and assign values based on the GP sample
    k = np.zeros((Nx, Ny))
    k_rand = np.random.permutation([k1, k2])  # Randomly assign k1 and k2
    for i in range(Nx):
        k[:, i] = np.where(i < ffn_norm, k_rand[0], k_rand[1])

    return x, ffn_norm, k


def fftIndgen(n):
    """
    Generate an array of indices for FFT transformation.

    Parameters:
    - n (int): Size of the array.

    Returns:
    - (list): List of indices for FFT.
    """
    a = range(0, int(n/2 + 1))
    b = [-i for i in reversed(range(1, int(n/2)))]
    return list(a) + b


def gaussian_random_field(Pk=lambda k: k**-3.0, size=100):
    """
    Generate a Gaussian random field using a power spectrum.

    Parameters:
    - Pk (callable): Power spectrum function.
    - size (int): Size of the generated field.

    Returns:
    - (ndarray): Gaussian random field.
    """
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))

    noise = np.fft.fft2(np.random.normal(size=(size, size))) 
    amplitude = np.zeros((size, size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = Pk2(kx, ky)

    return np.fft.ifft2(noise * amplitude).real

def rescale_linear(array, new_min, new_max):
    """
    Rescale an array linearly between new minimum and maximum values.

    Parameters:
    - array (ndarray): Input array to be rescaled.
    - new_min (float): New minimum value.
    - new_max (float): New maximum value.

    Returns:
    - (ndarray): Rescaled array.
    """
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    return m * array + b + new_min - m * minimum

def split_intervals(k, minK, maxK, n_intervals):
    """
    Split the values of an array into specified intervals.

    Parameters:
    - k (ndarray): Input array.
    - minK (float): Minimum value of the intervals.
    - maxK (float): Maximum value of the intervals.
    - n_intervals (int): Number of intervals.

    Returns:
    - (ndarray): Array 

    https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range """

    intervals = np.linspace(start=minK, stop=maxK, num=n_intervals+1)

    k = np.array(intervals)[np.searchsorted(intervals, k)]
    
    # one class has only one value: I remove it 
    unique , counts = np.unique(k, return_counts=True)
    d_unique = dict(zip(counts, unique))
    if 1 in d_unique: 
           i, j = np.where(k == d_unique.get(1))
           k[i, j] = k[i-1, j] 
    
    return np.round(k, 6) 

def reshuffle(k):
    """
    Randomly reshuffles the unique values in an array.

    Parameters:
    - k (ndarray): Array containing the values to be reshuffled.

    Returns:
    - (ndarray): Array with its unique values reshuffled.
    """
    unique_values = np.unique(k)
    shuffled_values = np.random.permutation(unique_values)
    reshuffled_map = dict(zip(unique_values, shuffled_values))
    return np.vectorize(reshuffled_map.get)(k)

def create_Krf(alpha=-7.0, discrete=False, minK=5, maxK=25, n_classes=5, size=32, reshuffle_k=True, log_shape=False):
    """
    Creates a hydraulic conductivity field based on Gaussian random fields.

    Parameters:
    - alpha (float): Exponent for the power spectrum function.
    - discrete (bool): Whether to discretize the field into classes.
    - minK, maxK (float): Minimum and maximum values for conductivity.
    - n_classes (int): Number of classes for discretization.
    - size (int): Size of the generated field.
    - reshuffle_k (bool): Whether to reshuffle the conductivity values.
    - log_shape (bool): Whether to apply logarithmic shaping to the field.

    Returns:
    - (ndarray): Generated hydraulic conductivity field.
    """
    field = gaussian_random_field(Pk=lambda k: k**alpha, size=size)
    k = field.real

    if log_shape:
        k = np.exp(k)

    k = rescale_linear(k, minK, maxK)

    if discrete:
        k = split_intervals(k, minK, maxK, n_classes)

    if reshuffle_k:
        k = reshuffle(k)

    return k

def generate_channelized_array(size, k1, k2):
    """
    Generates a channelized hydraulic conductivity field.

    Parameters:
    - size (int): The size of the generated field.
    - k1, k2 (float): Conductivity values for the two types of facies.

    Returns:
    - (ndarray): Channelized hydraulic conductivity field.
    """
    array = np.zeros((size, size))
    num_channels = 5
    channel_width = 2

    for _ in range(num_channels):
        start_point = np.random.randint(0, size)
        end_point = np.random.randint(0, size)
        amplitude = (end_point - start_point) / 2
        center = (end_point + start_point) / 2
        phase = np.random.rand() * np.pi * 2

        for x in range(size):
            y = int(amplitude * np.sin((2 * np.pi * x / size) + phase) + center)
            if 0 <= y < size:
                array[max(0, y - channel_width // 2):min(size, y + channel_width // 2 + 1), x] = k2

    array[array == 0] = k1
    return array

def ressample(arr, N):
    """Split array into N chunks """
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        block = np.hsplit(v, arr.shape[0] // N)
        A.extend([*block])
    return np.array(A)

def split_chunks(size = 32 , chunk_size = 4, mink= 5, maxk=25, log_intervals = False):
       """Split array into N chunks and assign random k to each chunk"""
       arr = np.zeros((size, size))  
       
       arr_reas =  ressample(arr, chunk_size) #--> chunk size 4
       
       #assign k to each chunk
       new_array = np.empty((size, size))
       count = 0
       for j in range(0, size, chunk_size):
              for i in range(0, size, chunk_size):
                     if log_intervals == False:
                        arr_reas[count] = random.randint(mink, maxk)
                     else:
                        arr_reas[count] = random.choice(np.geomspace(mink, maxk,num=100))
                        new_array[i:i+chunk_size, j:j+chunk_size] = arr_reas[count, :, :]
                        count += 1

       return new_array

def plot_3d(img):
    resolution=32
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')
#     ax = Axes3D(fig)

    x = np.linspace(0,1,resolution)
    y = np.linspace(0,1,resolution)

    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = img

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.cividis)

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()



# Main execution block for testing and visualizing the conductivity fields
if __name__ == '__main__':

           
    # Option:
    # - 'interface' : 2 materials split by an interface
    # - 'classes': a gaussian random field then discretized into classes
    # - 'chunks': a domain made up of 4x4 blocks of random material
    Option = 'classes'

      
    # Interface Option: Generates a discontinuity in a domain with 2 classes
    if Option=='interface':
        
              x, interface, k = k_discrete2(Nx = 32 , Ny = 32, k1 = 0.5 , k2 = 2.5)

              fig, (ax1, ax2) = plt.subplots(1,2)
              ax1.plot(interface, x)
              ax1.set_title(r"Interface")
              ax1.set_ylabel("y")
              ax1.set_xlabel("x")
              ax2.set_title('Discontinuous k field')
              im = ax2.imshow(k,  origin='lower', aspect="auto")
              fig.colorbar(im, ax=ax2, orientation='vertical')
              fig.show()
    # Classes or Chunks Options: Generates random fields or domain split into chunks
    else:
        # Visualization and plotting logic for the classes or chunks options
              a = 2  # number of rows
              b = 3  # number of columns
              c = 1  # initialize plot counter
              fig = plt.figure(figsize=(14,10))
              
              k_mid = []
              n_cl_l = []
              
              for realizations in range(6):
                 
                  plt.subplot(a, b, c)
                  if realizations < 3:
                         n_cl = np.random.randint( 2, 5 ) 
                         typeclass = 'Train'
                  else: 
                         n_cl = np.random.randint( 2, 5 )
                         typeclass = 'Test'
                         
                  plt.title('$n = {}$'.format(n_cl))
                  
                  if Option == 'classes':
                         k = create_Krf(discrete = False, minK = 0.0001 , maxK = 0.005, n_classes = n_cl, size = 32, reshuffle_k = False, log_shape = True)
                         # n_cl = 5 
                         # k1 = 0.0001
                         # k2 = 0.005
                         plot_3d(k)

                  elif Option=='chunks':
                         k = split_chunks(size = 32 , chunk_size = 4, mink= 0.0001, maxk=0.005, log_intervals = True)
                         
                  cmap = mpl.cm.viridis
                  classes = np.unique(k)
                  classes_b = classes
                  classes_b[0] -= 0.1
                  classes_b[0] += 0.1
                  # norm = mpl.colors.BoundaryNorm(list(classes), cmap.N)

                  im = plt.imshow(k , 
                           origin='lower', aspect='auto', cmap = cmap)
                  plt.colorbar(im, cmap=cmap,
                               ticks=classes, format='%1i')
                  c = c + 1
                  
                  k_mid.append(k[16])
                  n_cl_l.append(n_cl)     
              
              plt.show()
              
              fig = plt.figure()
              gs = fig.add_gridspec(2, hspace=0)
              axs = gs.subplots(sharex=True, sharey=True)
              for realizations in range(6):
                     ax_ind = 0 if realizations<3 else 1
                     axs[ax_ind].plot(k_mid[realizations], '-', label='$n = {}$'.format(n_cl_l[realizations] ))
              # Hide x labels and tick labels for all but bottom plot.
              for ax in axs:
                  ax.label_outer()
              axs[0].legend(loc="upper right")         
              axs[1].legend(loc="upper right")

              # Set common labels
              fig.text(0.5, 0.04, 'x [m]', ha='center', va='center')
              fig.text(0.06, 0.5, 'K [m/day]', ha='center', va='center', rotation='vertical')





