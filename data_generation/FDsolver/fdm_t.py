
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve # to use its short name
from collections import namedtuple
#import pdb

class InputError(Exception):
    pass

def unique(x, tol=0.0001):
    """return sorted unique values of x, keeping ascending or descending direction"""
    if x[0]>x[-1]:  # vector is reversed
        x = np.sort(x)[::-1]  # sort and reverse
        return x[np.hstack((np.diff(x) < -tol, True))]
    else:
        x = np.sort(x)
        return x[np.hstack((np.diff(x) > +tol, True))]


def fdm3t(gr, t, kxyz, Ss, FQ, HI, IBOUND, epsilon=0.67):
    '''Transient 3D Finite Difference Model returning computed heads and flows.

    Heads and flows are returned as 3D arrays as specified under output parmeters.

    Parameters
    ----------
    'gr' : `grid_object`, generated by gr = Grid(x, y, z, ..)
        if `gr.axial`==True, then the model is run in axially symmetric model
    t : ndarray, shape: [Nt+1]
        times at which the heads and flows are desired including the start time,
        which is usually zero, but can have any value.
    `kx`, `ky`, `kz` : ndarray, shape: (Ny, Nx, Nz), [L/T]
        hydraulic conductivities along the three axes, 3D arrays.
    `Ss` : ndarray, shape: (Ny, Nx, Nz), [L-1]
        specific elastic storage
    `FQ` : ndarray, shape: (Ny, Nx, Nz), [L3/T]
        prescrived cell flows (injection positive, zero of no inflow/outflow)
    `IH` : ndarray, shape: (Ny, Nx, Nz), [L]
        initial heads. `IH` has the prescribed heads for the cells with prescribed head.
    `IBOUND` : ndarray, shape: (Ny, Nx, Nz) of int
        boundary array like in MODFLOW with values denoting
        * IBOUND>0  the head in the corresponding cells will be computed
        * IBOUND=0  cells are inactive, will be given value NaN
        * IBOUND<0  coresponding cells have prescribed head
    `epsilon` : float, dimension [-]
        degree of implicitness, choose value between 0.5 and 1.0

    outputs
    -------
    `out` : namedtuple containing heads and flows:
        `out.Phi` : ndarray, shape: (Nt+1, Ny, Nx, Nz), [L3/T]
            computed heads. Inactive cells will have NaNs
            To get heads at time t[i], use Out.Phi[i]
            Out.Phi[0] = initial heads
        `out.Q`   : ndarray, shape: (Nt, Ny, Nx, Nz), [L3/T]
            net inflow in all cells during time step, inactive cells have 0
            Q during time step i, use Out.Q[i]
        `out.Qs`  : ndarray, shape: (Nt, Ny, Nx, Nz), [L3/T]
            release from storage during time step.
        `out.Qx   : ndarray, shape: (Nt, Ny, Nx-1, Nz), [L3/T]
            intercell flows in x-direction (parallel to the rows)
        `out.Qy`  : ndarray, shape: (Nt, Ny-1, Nx, Nz), [L3/T]
            intercell flows in y-direction (parallel to the columns)
        `out.Qz`  : ndarray, shape: (Nt, Ny, Nx, Nz-1), [L3/T]
            intercell flows in z-direction (vertially upward postitive)

    TO 161024
    '''

    if gr.axial:
        print('Running in axial mode, y-values are ignored.')

    kx, ky, kz = kxyz

    if kx.shape != gr.shape:
        raise AssertionError("shape of kx {0} differs from that of model {1}".format(kx.shape,gr.shape))
    if ky.shape != gr.shape:
        raise AssertionError("shape of ky {0} differs from that of model {1}".format(ky.shape,gr.shape))
    if kz.shape != gr.shape:
        raise AssertionError("shape of kz {0} differs from that of model {1}".format(kz.shape,gr.shape))
    if Ss.shape != gr.shape:
        raise AssertionError("shape of Ss {0} differs from that of model {1}".format(Ss.shape,gr.shape))

    active = (IBOUND>0).reshape(gr.Nod,)  # boolean vector denoting the active cells
    inact  = (IBOUND==0).reshape(gr.Nod,) # boolean vector denoting inacive cells
    fxhd   = (IBOUND<0).reshape(gr.Nod,)  # boolean vector denoting fixed-head cells

    # reshaping shorthands
    dx = np.reshape(gr.dx, (1, 1, gr.Nx))
    dy = np.reshape(gr.dy, (1, gr.Ny, 1))
    
    # half cell flow resistances
    if not gr.axial:
        Rx1 = 0.5 *    dx / (   dy * gr.DZ) / kx
        Rx2 = Rx1
        Ry1 = 0.5 *    dy / (gr.DZ *    dx) / ky
        Rz1 = 0.5 * gr.DZ / (   dx *    dy) / kz
    else:
        # prevent div by zero warning in next line; has not effect because x[0] is not used
        x = gr.x.copy();  x[0] = x[0] if x[0]>0 else 0.1* x[1]

        Rx1 = 1 / (2 * np.pi * kx * gr.DZ) * np.log(x[1:] /  gr.xm).reshape((1, 1, gr.Nx))
        Rx2 = 1 / (2 * np.pi * kx * gr.DZ) * np.log(gr.xm / x[:-1]).reshape((1, 1, gr.Nx))
        Ry1 = np.inf * np.ones(gr.shape)
        Rz1 = 0.5 * gr.DZ / (np.pi * (gr.x[1:]**2 - gr.x[:-1]**2).reshape((1, 1, gr.Nx)) * kz)
        
    


    # set flow resistance in inactive cells to infinite
    Rx1[inact.reshape(gr.shape)] = np.inf
    Rx2[inact.reshape(gr.shape)] = np.inf
    Ry1[inact.reshape(gr.shape)] = np.inf
    Ry2 = Ry1
    Rz1[inact.reshape(gr.shape)] = np.inf
    Rz2 = Rz1

    # conductances between adjacent cells
    Cx = 1 / (Rx1[  :, :,  :-1] + Rx2[:, : , 1:])        
    Cy = 1 / (Ry1[  :, :-1,:  ] + Ry2[:, 1:, : ])
    Cz = 1 / (Rz1[:-1, :,  :  ] + Rz2[1:, :, : ])

    # storage term, variable dt not included
    Cs = (Ss * gr.Volume / epsilon).ravel()

    # cell number of neighboring cells
    IE = gr.NOD[  :, :,   1:] # east neighbor cell numbers
    IW = gr.NOD[  :, :,  :-1] # west neighbor cell numbers
    IN = gr.NOD[  :, :-1,  :] # north neighbor cell numbers
    IS = gr.NOD[  :, 1:,   :] # south neighbor cell numbers
    IT = gr.NOD[:-1, :,    :] # top neighbor cell numbers
    IB = gr.NOD[ 1:, :,    :] # bottom neighbor cell numbers

    R = lambda x : x.ravel()  # generate anonymous function R(x) as shorthand for x.ravel()

    # notice the call  csc_matrix( (data, (rowind, coind) ), (M,N))  tuple within tupple
    # also notice that Cij = negative but that Cii will be postive, namely -sum(Cij)
    A = sp.csc_matrix(( np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz)) ),\
                        (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IB), R(IT)) ),\
                         np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)) ),\
                      )),(gr.Nod,gr.Nod))

    A = -A + sp.diags(np.array(A.sum(axis=1))[:,0]) # Change sign and add diagonal

    #Initialize output arrays (= memory allocation)
    Nt = len(t)-1
    Phi = np.zeros((Nt+1, gr.Nod)) # Nt+1 times
    Q   = np.zeros((Nt  , gr.Nod)) # Nt time steps
    Qs  = np.zeros((Nt  , gr.Nod))
    Qx  = np.zeros((Nt, gr.Nz, gr.Ny, gr.Nx-1))
    Qy  = np.zeros((Nt, gr.Nz, gr.Ny-1, gr.Nx))
    Qz  = np.zeros((Nt, gr.Nz-1, gr.Ny, gr.Nx))
    
    # reshape input arrays to vectors for use in system equation
    FQ = R(FQ);  HI = R(HI);  Cs = R(Cs)

    # initialize heads
    Phi[0] = HI

    # solve heads at active locations at t_i+eps*dt_i

    Nt=len(t)  # for heads, at all times Phi at t[0] = initial head
    Ndt=len(np.diff(t)) # for flows, average within time step

    for idt, dt in enumerate(np.diff(t)):

        it = idt+1

        # this A is not complete !!
        RHS = FQ - (A + sp.diags(Cs / dt))[:,fxhd].dot(Phi[it-1][fxhd]) # Right-hand side vector

        Phi[it][active] = spsolve( (A + sp.diags(Cs / dt))[active][:,active],
                                  RHS[active] + Cs[active] / dt * Phi[it-1][active])

        # net cell inflow
        Q[idt]  = A.dot(Phi[it])

        Qs[idt] = -Cs/dt * (Phi[it]-Phi[it-1])


        #Flows across cell faces
        Qx[idt] =  -np.diff( Phi[it].reshape(gr.shape), axis=2) * Cx
        Qy[idt] =  +np.diff( Phi[it].reshape(gr.shape), axis=1) * Cy
        Qz[idt] =  +np.diff( Phi[it].reshape(gr.shape), axis=0) * Cz

        # update head to end of time step
        Phi[it] = Phi[it-1] + (Phi[it]-Phi[it-1])/epsilon

    # reshape Phi to shape of grid
    Phi = Phi.reshape((Nt,) + gr.shape)
    Q   = Q.reshape( (Ndt,) + gr.shape)
    Qs  = Qs.reshape((Ndt,) + gr.shape)

    Out = namedtuple('Out',['t', 'Phi', 'Q', 'Qs', 'Qx', 'Qy', 'Qz'])
    Out.__doc__ = """fdm3 output, <namedtuple>, containing fields
                    `t`, `Phi`, `Q`, `Qs`, `Qx`, `Qy` and `Qz`\n \
                    Use Out.Phi, Out.Q, Out.Qx, Out.Qy and Out.Qz
                    or
                    Out.Phi[i] for the 3D heads of time `i`
                    Out.Q[i] for the 3D flows of time step `i`
                    Notice the difference between time and time step
                    The shape of Phi is (Nt + 1,Ny, Nx, Nz)
                    The shape of Q, Qs is (Nt, Ny, Nx, Nz)
                    For the other shapes see docstring of fdm_t
                    """


    out = Out(t=t, Phi=Phi, Q=Q, Qs=Qs, Qx=Qx, Qy=Qy, Qz=Qz )

    return out # all outputs in a named tuple for easy access