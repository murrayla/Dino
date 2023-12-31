import numpy as np
import sympy as sym
import multiprocessing as mp
import scipy as sp
from functools import partial
from matplotlib.patches import Polygon
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('dark_background')

FILE_NAME = "3Dtwotet"
N_EL_N = 10
DIM    = 3
E   = 200 
NU  = 0.20
XI   = sym.Symbol('xi', real=True)
ETA  = sym.Symbol('eta', real=True)
ZETA = sym.Symbol('zeta', real=True)
WE = np.array([-4/5, 9/20, 9/20, 9/20, 9/20])
GP = np.array([[1/4, 1/4, 1/4], [1/2, 1/6, 1/6], 
               [1/6, 1/2, 1/6], [1/6, 1/6, 1/2],
               [1/6, 1/6, 1/6]])
beta = 1 - XI - ETA - ZETA # Dependent

# Individual functions
# Corners
n1  = XI*(2*XI-1)        
n2  = ETA*(2*ETA-1)
n3  = ZETA*(2*ZETA-1)
n4  = beta*(2*beta-1)
# Edges
n5  = 4*XI*ETA
n6  = 4*ETA*ZETA
n7  = 4*ZETA*XI
n8  = 4*XI*beta
n9  = 4*ZETA*beta
n10 = 4*beta*ETA

# Function matrix
phi = sym.Matrix([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10])

# Derivative of Shape Functions
# delPhi = [δφ1/δξ δφ2/δξ ... δφ10/δξ
#           δφ1/δξ δφ2/δξ ... δφ10/δξ
#           δφ1/δζ δφ2/δζ ... δφ10/δζ]
delPhi = sym.Matrix([[sym.diff(phi[j], XI, 1)   for j in range(0, N_EL_N, 1)],  
                        [sym.diff(phi[j], ETA, 1)  for j in range(0, N_EL_N, 1)],
                        [sym.diff(phi[j], ZETA, 1) for j in range(0, N_EL_N, 1)]])

def calc_k_laplace(ele, np_nodes, np_eles, phi, delPhi, d, xi, eta, zeta, dim, n_el_n, we, gp):
    
    n_n = int(len(np_nodes[:, 0]))

    ## STIFFNESS MATRIX
    # Preallocate Matrices
    k_gl = np.zeros([dim*n_n, dim*n_n])

    # Preallocate natural coordinates
    coords = np.zeros((n_el_n, dim))

    # Initialise K_local
    k_lo = np.zeros((dim * n_el_n, dim * n_el_n))

    # Set rows and columns GLOBAL
    rc = np_eles[ele, :]

    # Find coordinates for those nodes
    np_nodes_idx = np_nodes[:, 0]
    for i, local_node in enumerate(rc):
        node_indices = np.where(np_nodes_idx == local_node)[0]
        coords[i] = np_nodes[node_indices, 1:dim+1][0]

    # Jacobian
    jac   = delPhi * coords
    detJ  = jac.det()
    b_sub = jac.inv() * delPhi

    # B Matrix
    b_mat = sym.zeros(6,dim*n_el_n)
    for r, c in enumerate(range(0, dim*n_el_n, dim)):
        # [δφ/δx,      0,      0] 
        b_mat[0, c]   = b_sub[0, r]
        # [    0,  δφ/δy,      0]
        b_mat[1, c+1] = b_sub[1, r]
        # [    0,      0,  δφ/δz]
        b_mat[2, c+2] = b_sub[2, r]
    
    # Stiffness Matrix, K = B'DB
    k_stiff = b_mat.T*b_mat*detJ
    
    # Iterate through nodes LOCAL
    for i in range(0, n_el_n, 1):
        for j in range(0, n_el_n, 1):

            # Gauss Quadrature
            for q, w in enumerate(we):
                 # [1 2 3]
                k_lo[dim*i+0, dim*j+0]   = k_lo[dim*i+0, dim*j+0] + w *  float(k_stiff[dim*i+0, dim*j+0].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+0, dim*j+1]   = k_lo[dim*i+0, dim*j+1] + w *  float(k_stiff[dim*i+0, dim*j+1].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+0, dim*j+2]   = k_lo[dim*i+0, dim*j+2] + w *  float(k_stiff[dim*i+0, dim*j+2].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                # [4 5 6]
                k_lo[dim*i+1, dim*j+0]   = k_lo[dim*i+1, dim*j+0] + w *  float(k_stiff[dim*i+1, dim*j+0].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+1, dim*j+1]   = k_lo[dim*i+1, dim*j+1] + w *  float(k_stiff[dim*i+1, dim*j+1].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+1, dim*j+2]   = k_lo[dim*i+1, dim*j+2] + w *  float(k_stiff[dim*i+1, dim*j+2].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                # [7 8 9]
                k_lo[dim*i+2, dim*j+0]   = k_lo[dim*i+2, dim*j+0] + w *  float(k_stiff[dim*i+2, dim*j+0].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+2, dim*j+1]   = k_lo[dim*i+2, dim*j+1] + w *  float(k_stiff[dim*i+2, dim*j+1].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+2, dim*j+2]   = k_lo[dim*i+2, dim*j+2] + w *  float(k_stiff[dim*i+2, dim*j+2].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))

            # Store global
            # [1 2 3]
            k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+0] = k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+0] + k_lo[dim*i+0, dim*j+0] 
            k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+1] = k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+1] + k_lo[dim*i+0, dim*j+1] 
            k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+2] = k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+2] + k_lo[dim*i+0, dim*j+2] 
            # [4 5 6]
            k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+0] = k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+0] + k_lo[dim*i+1, dim*j+0] 
            k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+1] = k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+1] + k_lo[dim*i+1, dim*j+1] 
            k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+2] = k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+2] + k_lo[dim*i+1, dim*j+2] 
            # [7 8 9]
            k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+0] = k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+0] + k_lo[dim*i+2, dim*j+0] 
            k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+1] = k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+1] + k_lo[dim*i+2, dim*j+1] 
            k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+2] = k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+2] + k_lo[dim*i+2, dim*j+2] 

    print("CALCULATIONS ON ELEMENT {} COMPLETE".format(ele))
    return k_gl

def element_assign(el_type, el_order):
    # Triangle
    if el_type == 0:
        # Linear
        if el_order == 0:
            delPhi = 0
        # Quadratic
        elif el_order == 1:
            delPhi = 1
        # Cubic
        elif el_order == 2:
            delPhi = 1
        # Other
        else: 
            return None
    # Tetrahedron
    elif el_type == 1:
        dim = 3
        # Linear
        if el_order == 0:
            delPhi = 0
        # Quadratic
        elif el_order == 1:
            n_el_n = 10
            xi = sym.Symbol('xi', real=True)
            eta = sym.Symbol('eta', real=True)
            zeta = sym.Symbol('zeta', real=True)
            beta = 1 - xi - eta - zeta # Dependent
            # Individual functions
            # Corners
            n1  = xi*(2*xi-1)        
            n2  = eta*(2*eta-1)
            n3  = zeta*(2*zeta-1)
            n4  = beta*(2*beta-1)
            # Edges
            n5  = 4*xi*eta
            n6  = 4*eta*zeta
            n7  = 4*zeta*xi
            n8  = 4*xi*beta
            n9  = 4*zeta*beta
            n10 = 4*beta*eta
            # Shape Functions
            phi = sym.Matrix([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10])
            # Derivative of Shape Functions
            # delPhi = [δφ1/δξ δφ2/δξ ... δφ10/δξ
            #           δφ1/δξ δφ2/δξ ... δφ10/δξ
            #           δφ1/δζ δφ2/δζ ... δφ10/δζ]
            delPhi = sym.Matrix([[sym.diff(phi[j], xi, 1)   for j in range(0, n_el_n, 1)],  
                        [sym.diff(phi[j],eta, 1)  for j in range(0, n_el_n, 1)],
                        [sym.diff(phi[j], zeta, 1) for j in range(0, n_el_n, 1)]]
                        )
            return dim, n_el_n, (xi, eta, zeta), phi, delPhi
        # Cubic
        elif el_order == 2:
            delPhi = 1
        # Other
        else: 
            return None
    # Rectangle
    elif el_type == 2:
        # Linear
        if el_order == 0:
            delPhi = 0
        # Quadratic
        elif el_order == 1:
            delPhi = 1
        # Cubic
        elif el_order == 2:
            delPhi = 1
        # Other
        else: 
            return None
    # Hexahedron
    elif el_type == 3:
        # Linear
        if el_order == 0:
            delPhi = 0
        # Quadratic
        elif el_order == 1:
            delPhi = 1
        # Cubic
        elif el_order == 2:
            delPhi = 1
        # Other
        else: 
            return None
    # Other
    else:
        return None

def calc_k_linear_elasticity(ele, np_nodes, np_eles, phi, delPhi, d, xi, eta, zeta, dim, n_el_n, we, gp):

    n_n = int(len(np_nodes[:, 0]))

    ## STIFFNESS MATRIX
    # Preallocate Matrices
    k_gl = np.zeros([dim*n_n, dim*n_n])

    # Preallocate natural coordinates
    coords = np.zeros((n_el_n, dim))

    # Initialise K_local
    k_lo = np.zeros((dim * n_el_n, dim * n_el_n))

    # Set rows and columns GLOBAL
    rc = np_eles[ele, :]

    # Find coordinates for those nodes
    np_nodes_idx = np_nodes[:, 0]
    for i, local_node in enumerate(rc):
        node_indices = np.where(np_nodes_idx == local_node)[0]
        coords[i] = np_nodes[node_indices, 1:dim+1][0]

    # Jacobian
    jac   = delPhi * coords
    detJ  = jac.det()
    b_sub = jac.inv() * delPhi

    # B Matrix
    b_mat = sym.zeros(6,dim*n_el_n)
    for r, c in enumerate(range(0, dim*n_el_n, dim)):
        # [δφ/δx,      0,      0] 
        b_mat[0, c]   = b_sub[0, r]
        # [    0,  δφ/δy,      0]
        b_mat[1, c+1] = b_sub[1, r]
        # [    0,      0,  δφ/δz]
        b_mat[2, c+2] = b_sub[2, r]
        # [δφ/δy,  δφ/δx,      0]
        b_mat[3, c]   = b_sub[1, r]
        b_mat[3, c+1] = b_sub[0, r]
        # [    0,  δφ/δz,  δφ/δy]
        b_mat[4, c+1] = b_sub[2, r]
        b_mat[4, c+2] = b_sub[1, r]
        # [δφ/δz,      0,  δφ/δx]  
        b_mat[5, c]   = b_sub[2, r]
        b_mat[5, c+2] = b_sub[0, r] 
    
    # Stiffness Matrix, K = B'DB
    k_stiff = b_mat.T*d*b_mat*detJ
    
    # Iterate through nodes LOCAL
    for i in range(0, n_el_n, 1):
        for j in range(0, n_el_n, 1):

            # Gauss Quadrature
            for q, w in enumerate(we):
                 # [1 2 3]
                k_lo[dim*i+0, dim*j+0]   = k_lo[dim*i+0, dim*j+0] + w *  float(k_stiff[dim*i+0, dim*j+0].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+0, dim*j+1]   = k_lo[dim*i+0, dim*j+1] + w *  float(k_stiff[dim*i+0, dim*j+1].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+0, dim*j+2]   = k_lo[dim*i+0, dim*j+2] + w *  float(k_stiff[dim*i+0, dim*j+2].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                # [4 5 6]
                k_lo[dim*i+1, dim*j+0]   = k_lo[dim*i+1, dim*j+0] + w *  float(k_stiff[dim*i+1, dim*j+0].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+1, dim*j+1]   = k_lo[dim*i+1, dim*j+1] + w *  float(k_stiff[dim*i+1, dim*j+1].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+1, dim*j+2]   = k_lo[dim*i+1, dim*j+2] + w *  float(k_stiff[dim*i+1, dim*j+2].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                # [7 8 9]
                k_lo[dim*i+2, dim*j+0]   = k_lo[dim*i+2, dim*j+0] + w *  float(k_stiff[dim*i+2, dim*j+0].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+2, dim*j+1]   = k_lo[dim*i+2, dim*j+1] + w *  float(k_stiff[dim*i+2, dim*j+1].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))
                k_lo[dim*i+2, dim*j+2]   = k_lo[dim*i+2, dim*j+2] + w *  float(k_stiff[dim*i+2, dim*j+2].subs({xi: gp[q, 0],eta: gp[q, 1], zeta: gp[q, 2]}))

            # Store global
            # [1 2 3]
            k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+0] = k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+0] + k_lo[dim*i+0, dim*j+0] 
            k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+1] = k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+1] + k_lo[dim*i+0, dim*j+1] 
            k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+2] = k_gl[dim*(rc[i]-1)+0, dim*(rc[j]-1)+2] + k_lo[dim*i+0, dim*j+2] 
            # [4 5 6]
            k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+0] = k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+0] + k_lo[dim*i+1, dim*j+0] 
            k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+1] = k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+1] + k_lo[dim*i+1, dim*j+1] 
            k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+2] = k_gl[dim*(rc[i]-1)+1, dim*(rc[j]-1)+2] + k_lo[dim*i+1, dim*j+2] 
            # [7 8 9]
            k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+0] = k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+0] + k_lo[dim*i+2, dim*j+0] 
            k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+1] = k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+1] + k_lo[dim*i+2, dim*j+1] 
            k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+2] = k_gl[dim*(rc[i]-1)+2, dim*(rc[j]-1)+2] + k_lo[dim*i+2, dim*j+2] 

    print("CALCULATIONS ON ELEMENT {} COMPLETE".format(ele))
    return k_gl

def gauss_num_int(el_type, order):

    if el_type == 0:
        p = 0
    elif el_type == 1:
        if order == 5:
            we = np.array([-4/5, 9/20, 9/20, 9/20, 9/20])
            gp = np.array([[1/4, 1/4, 1/4], [1/2, 1/6, 1/6], 
                        [1/6, 1/2, 1/6], [1/6, 1/6, 1/2],
                        [1/6, 1/6, 1/6]])
            return we, gp
        elif order == 1:
            return None
        else:
            return None
    else:
        return None

def rotation_matrix(angle):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # Define the rotation matrix
        # rotation_matrix = np.array([[1, 0, 0],
        #                             [0, cos_theta, -sin_theta],
        #                             [0, sin_theta, cos_theta]])
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                    [cos_theta,  sin_theta, 0],
                                    [0, 0, 1]])
        
        return rotation_matrix

def get_D_mat(Em, nu):
    mu  = Em/(2*(1+nu))
    lam = nu*Em/((1+nu)*(1-2*nu))

    # Constitutive Matrix
    # σ = Dε = D∇u
    # [sigma_xx]   [lam + 2 * mu, lam, lam, 0, 0, 0]   [epsilon_xx]
    # [sigma_yy]   [lam, lam + 2 * mu, lam, 0, 0, 0]   [epsilon_yy]
    # [sigma_zz] = [lam, lam, lam + 2 * mu, 0, 0, 0] * [epsilon_zz]
    # [sigma_xy]   [  0,   0,   0,   mu,   0,   0  ]   [epsilon_xy]
    # [sigma_yz]   [  0,   0,   0,   0,   mu,   0  ]   [epsilon_yz]
    # [sigma_zx]   [  0,   0,   0,   0,    0,  mu  ]   [epsilon_zx]
    d = np.array([[lam + 2 * mu, lam, lam, 0, 0, 0],
                 [lam, lam + 2 * mu, lam, 0, 0, 0],
                 [lam, lam, lam + 2 * mu, 0, 0, 0],
                 [  0,   0,   0,   mu,   0,   0  ],
                 [  0,   0,   0,   0,   mu,   0  ],
                 [  0,   0,   0,   0,   0,   mu  ]])
    return d, mu, lam

def apply_BC_3D(np_nodes, k_gl_sol, f_gl_sol, BC_0, BC_1, axi):

    dim     = 3
    min_val = np.amin(np_nodes[:, axi+1])
    max_val = np.amax(np_nodes[:, axi+1])

    # Apply Bounadry Conditions
    # Check position of node and apply BC if condition met
    for n in np_nodes[:, 0]:
        n_idx = int(n)
        n_val = np_nodes[np_nodes[:, 0] == n, axi+1][0]

        if n_val == min_val:
            f_gl_sol[dim*(n_idx-1)+axi]                   = BC_0
            k_gl_sol[dim*(n_idx-1)+axi, :]                = 0
            k_gl_sol[dim*(n_idx-1)+axi,dim*(n_idx-1)+axi] = 1

        elif n_val == max_val:
            f_gl_sol[dim*(n_idx-1)+axi]                   = BC_1
            k_gl_sol[dim*(n_idx-1)+axi, :]                = 0
            k_gl_sol[dim*(n_idx-1)+axi,dim*(n_idx-1)+axi] = 1

    return k_gl_sol, f_gl_sol

def apply_torsion_3D(np_nodes, k_gl_sol, f_gl_sol, turn, axi):

    dim     = 3
    min_val = np.amin(np_nodes[:, axi+1])
    max_val = np.amax(np_nodes[:, axi+1])

    r_matrix = rotation_matrix(turn)

    # Apply Bounadry Conditions
    # Check position of node and apply BC if condition met
    for n in np_nodes[:, 0]:
        n_idx = int(n)
        n_val = np_nodes[np_nodes[:, 0] == n, axi+1][0]

        if n_val == min_val:
            # x
            f_gl_sol[dim*(n_idx-1)+0]                     = 0
            k_gl_sol[dim*(n_idx-1)+0, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+0,dim*(n_idx-1)+0]     = 1
            # y
            f_gl_sol[dim*(n_idx-1)+1]                     = 0
            k_gl_sol[dim*(n_idx-1)+1, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+1,dim*(n_idx-1)+1]     = 1
            # z
            f_gl_sol[dim*(n_idx-1)+2]                     = 0
            k_gl_sol[dim*(n_idx-1)+2, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+2,dim*(n_idx-1)+2]     = 1

        elif n_val == max_val:
            coord = np.array(np_nodes[np_nodes[:, 0] == n, 1:])
            rot_point = np.matmul(r_matrix, coord.T)
            # x
            f_gl_sol[dim*(n_idx-1)+0]                     = rot_point[0]
            k_gl_sol[dim*(n_idx-1)+0, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+0,dim*(n_idx-1)+0]     = 1
            # y
            f_gl_sol[dim*(n_idx-1)+1]                     = rot_point[1]
            k_gl_sol[dim*(n_idx-1)+1, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+1,dim*(n_idx-1)+1]     = 1
            # z
            f_gl_sol[dim*(n_idx-1)+2]                     = rot_point[2]
            k_gl_sol[dim*(n_idx-1)+2, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+2,dim*(n_idx-1)+2]     = 1

    return k_gl_sol, f_gl_sol

def apply_extension_3D(np_nodes, k_gl_sol, f_gl_sol, disp, axi):

    dim     = 3
    min_val = np.amin(np_nodes[:, axi+1])
    max_val = np.amax(np_nodes[:, axi+1])

    # Apply Bounadry Conditions
    # Check position of node and apply BC if condition met
    for n in np_nodes[:, 0]:
        n_idx = int(n)
        n_val = np_nodes[np_nodes[:, 0] == n, axi+1][0]

        if n_val == min_val:
            # x
            f_gl_sol[dim*(n_idx-1)+0]                     = 0
            k_gl_sol[dim*(n_idx-1)+0, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+0,dim*(n_idx-1)+0]     = 1
            # y
            f_gl_sol[dim*(n_idx-1)+1]                     = 0
            k_gl_sol[dim*(n_idx-1)+1, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+1,dim*(n_idx-1)+1]     = 1
            # z
            f_gl_sol[dim*(n_idx-1)+2]                     = 0
            k_gl_sol[dim*(n_idx-1)+2, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+2,dim*(n_idx-1)+2]     = 1

        elif n_val == max_val:
            # x
            f_gl_sol[dim*(n_idx-1)+0]                     = disp
            k_gl_sol[dim*(n_idx-1)+0, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+0,dim*(n_idx-1)+0]     = 1
            # y
            f_gl_sol[dim*(n_idx-1)+1]                     = 0
            k_gl_sol[dim*(n_idx-1)+1, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+1,dim*(n_idx-1)+1]     = 1
            # z
            f_gl_sol[dim*(n_idx-1)+2]                     = 0
            k_gl_sol[dim*(n_idx-1)+2, :]                  = 0
            k_gl_sol[dim*(n_idx-1)+2,dim*(n_idx-1)+2]     = 1

    return k_gl_sol, f_gl_sol

def epsSigU(ele, dim, xi, eta, zeta, np_eles, np_nodes, gp, u, phi, delPhi, mu, lam):

    n_el_n = len(np_eles[0, :])
    n_ele  = len(np_eles[:, 0])

    # GP Global stress matrices 
    sig_xx_gp   = np.zeros(n_ele*len(gp))
    sig_yy_gp   = np.zeros(n_ele*len(gp))
    sig_zz_gp   = np.zeros(n_ele*len(gp))

    # GP Global strain matrices 
    # norm
    eps_xx_gp   = np.zeros(n_ele*len(gp))
    eps_yy_gp   = np.zeros(n_ele*len(gp))
    eps_zz_gp   = np.zeros(n_ele*len(gp))
    # 
    eps_xy_gp   = np.zeros(n_ele*len(gp))
    eps_yz_gp   = np.zeros(n_ele*len(gp))
    eps_zx_gp   = np.zeros(n_ele*len(gp))
    eps_yx_gp   = np.zeros(n_ele*len(gp))
    eps_zy_gp   = np.zeros(n_ele*len(gp))
    eps_xz_gp   = np.zeros(n_ele*len(gp))

    # GP Global coordinates matrices
    coords_x_gp = np.zeros(n_ele*len(gp))
    coords_y_gp = np.zeros(n_ele*len(gp))
    coords_z_gp = np.zeros(n_ele*len(gp))

    udcoords_x_gp = np.zeros(n_ele*len(gp))
    udcoords_y_gp = np.zeros(n_ele*len(gp))
    udcoords_z_gp = np.zeros(n_ele*len(gp))

    # GP Global displacement matrices
    disp_u_gp   = np.zeros(n_ele*len(gp))
    disp_v_gp   = np.zeros(n_ele*len(gp))
    disp_w_gp   = np.zeros(n_ele*len(gp))

    # Store disp vals
    disp_vals = sym.zeros(n_el_n, dim)

    # Store natural coordinates
    coords = sym.zeros(n_el_n, dim)

    # Set rows and columns GLOBAL
    rc = np_eles[ele, :]

    # Find coordinates and displacements
    for i, local_node in enumerate(rc):
        disp_vals[i, 0] = u[(rc[i]-1)*dim]
        disp_vals[i, 1] = u[(rc[i]-1)*dim+1]
        disp_vals[i, 2] = u[(rc[i]-1)*dim+2]
        coords[i, 0]    = np_nodes[np.where(np_nodes[:, 0] == local_node), 1][0]
        coords[i, 1]    = np_nodes[np.where(np_nodes[:, 0] == local_node), 2][0]
        coords[i, 2]    = np_nodes[np.where(np_nodes[:, 0] == local_node), 3][0]
    
    # Set x and y, u and v, in terms of shape functions
    x    = phi.T * coords[:, 0]
    y    = phi.T * coords[:, 1]
    z    = phi.T * coords[:, 2]

    u_sf = phi.T * disp_vals[:, 0]
    v_sf = phi.T * disp_vals[:, 1]
    w_sf = phi.T * disp_vals[:, 2]
    
    # Determine δ{xyz}/δ{xietazeta} Jacobians and inverse
    jac_xyz     = delPhi * coords
    inv_jac_xyz = jac_xyz.inv()

    # Determine δ{uvw}/δ{xietazeta} Jacboain and δ{uvw}/δ{xyz}
    jac_uvw    = delPhi      * disp_vals
    duvwdxyz   = inv_jac_xyz * jac_uvw

# Gauss Quadrature 
    for q, (p1, p2, p3) in enumerate(gp):

        # Determine strains
        # [xx, xy, xz]
        eps_xx_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[0, 0].subs({xi: p1, eta: p2, zeta: p3}))
        eps_xy_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[0, 1].subs({xi: p1, eta: p2, zeta: p3}))
        eps_xz_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[0, 2].subs({xi: p1, eta: p2, zeta: p3}))
        # [yx, yy, yz]
        eps_yx_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[1, 0].subs({xi: p1, eta: p2, zeta: p3}))
        eps_yy_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[1, 1].subs({xi: p1, eta: p2, zeta: p3}))
        eps_yz_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[1, 2].subs({xi: p1, eta: p2, zeta: p3}))
        # [zx, zy, zz]
        eps_zx_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[2, 0].subs({xi: p1, eta: p2, zeta: p3}))
        eps_zy_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[2, 1].subs({xi: p1, eta: p2, zeta: p3}))
        eps_zz_gp[ele*len(gp)-1 + q]   = float(duvwdxyz[2, 2].subs({xi: p1, eta: p2, zeta: p3}))

        # Determine coordinates 
        coords_x_gp[ele*len(gp)-1 + q] = float((u_sf[0]+x[0]).subs({xi: p1, eta: p2, zeta: p3}))
        coords_y_gp[ele*len(gp)-1 + q] = float((v_sf[0]+y[0]).subs({xi: p1, eta: p2, zeta: p3}))
        coords_z_gp[ele*len(gp)-1 + q] = float((w_sf[0]+z[0]).subs({xi: p1, eta: p2, zeta: p3}))

        udcoords_x_gp[ele*len(gp)-1 + q] = float((x[0]).subs({xi: p1, eta: p2, zeta: p3}))
        udcoords_y_gp[ele*len(gp)-1 + q] = float((y[0]).subs({xi: p1, eta: p2, zeta: p3}))
        udcoords_z_gp[ele*len(gp)-1 + q] = float((z[0]).subs({xi: p1, eta: p2, zeta: p3}))

        # Determine displacements
        disp_u_gp[ele*len(gp)-1 + q]   = float(u_sf[0].subs({xi: p1, eta: p2, zeta: p3}))
        disp_v_gp[ele*len(gp)-1 + q]   = float(v_sf[0].subs({xi: p1, eta: p2, zeta: p3}))
        disp_w_gp[ele*len(gp)-1 + q]   = float(w_sf[0].subs({xi: p1, eta: p2, zeta: p3}))

        # Determine stress
        sig_xx_gp[ele*len(gp)-1 + q]   = 1000 * (2 * mu + lam) * eps_xx_gp[ele*len(gp)-1 + q] \
                                                        + lam * eps_yy_gp[ele*len(gp)-1 + q] \
                                                        + lam * eps_zz_gp[ele*len(gp)-1 + q]
        sig_yy_gp[ele*len(gp)-1 + q]   = 1000 * (2 * mu + lam) * eps_yy_gp[ele*len(gp)-1 + q] \
                                                        + lam * eps_xx_gp[ele*len(gp)-1 + q] \
                                                        + lam * eps_zz_gp[ele*len(gp)-1 + q]
        sig_zz_gp[ele*len(gp)-1 + q]   = 1000 * (2 * mu + lam) * eps_zz_gp[ele*len(gp)-1 + q] \
                                                        + lam * eps_yy_gp[ele*len(gp)-1 + q] \
                                                        + lam * eps_xx_gp[ele*len(gp)-1 + q]

    stress = np.array([sig_xx_gp, sig_yy_gp, sig_zz_gp]) 
    strain = np.array([eps_xx_gp, eps_yy_gp, eps_zz_gp]) 
    disp   = np.array([disp_u_gp, disp_v_gp, disp_w_gp]) 
    coords = np.array([coords_x_gp, coords_y_gp, coords_z_gp]) 
    undefc = np.array([udcoords_x_gp, udcoords_y_gp, udcoords_z_gp]) 

    print("ANALYSIS ON ELEMENT {} COMPLETE".format(ele))

    return stress, strain, disp, coords, undefc

def post_calc(k_gl, np_nodes, np_eles, phi, delPhi):

    n_n = int(len(np_nodes[:, 0]))
    f_gl = np.zeros(DIM*n_n)

    _, mu, lam = get_D_mat(E, NU)

    # Store K_gl_r
    k_gl_sol = np.copy(k_gl)
    f_gl_sol = np.copy(f_gl)
    u_sol    = np.zeros(DIM*n_n)

    # Displacement
    per_disp = 20
    disp     = per_disp/100*(np.amax(np_nodes[:, 1]) - np.amin(np_nodes[:, 1]))

    x0_BC = 0
    y0_BC = 0
    z0_BC = 0

    x1_BC = 0
    y1_BC = 0
    z1_BC = disp

    # Boundary Conditions
    k_gl_sol, f_gl_sol = apply_extension_3D(np_nodes, k_gl_sol, f_gl_sol, disp, 0)

    ## Displacements
    rhs = f_gl_sol #- np.matmul(k_gl, u_sol)
    u   = np.matmul(np.linalg.inv(k_gl_sol), rhs)
    f   = np.matmul(k_gl, u)

    np.savetxt("u_" + FILE_NAME + ".text", u, fmt='%f')
    np.savetxt("f_" + FILE_NAME + ".text", f, fmt='%f')

    _, _, disp, coords = epsSigU(FILE_NAME, np_eles, np_nodes, GP, u, phi, delPhi, mu, lam)

    plot_disps(u, np_nodes, disp, coords, np_eles)

def plot_complex_polygon(ax, vertices):
    polygon = Poly3DCollection([vertices], edgecolors='k', alpha=0.25)
    ax.add_collection3d(polygon)

def plot_disps(u, np_nodes, disp, coords, np_eles):
    dim = 3
    n_n    = int(len(np_nodes[:, 0]))

    # Displacement component matrices
    u_x = np.zeros([n_n, 1])
    u_y = np.zeros([n_n, 1])
    u_z = np.zeros([n_n, 1])

    # Node position matrices
    be_def_coord = np_nodes[:, 1:]
    af_def_coord = np.zeros([n_n, 3])

    # Mapping node positions and displacements
    for z, _ in enumerate(np_nodes[:, 0]):
        u_x[z] = u[dim * z]
        u_y[z] = u[dim * z + 1]
        u_z[z] = u[dim * z + 2]
        af_def_coord[z, 0] = be_def_coord[z, 0] + u[dim * z]
        af_def_coord[z, 1] = be_def_coord[z, 1] + u[dim * z + 1]
        af_def_coord[z, 2] = be_def_coord[z, 2] + u[dim * z + 2]

    # Set figures
    fig0 = plt.figure()

    # format
    cmap = get_cmap('Blues')
    ax = fig0.add_subplot(111, projection='3d')

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 6), (6, 7), (7, 9), (9, 2),
             (7, 8), (8, 4)]
    
    # Plot patches
    for _, el_ns in enumerate(np_eles):
        vertices = []
        # face = []
        curr_nodes = [el_ns[0]-1, el_ns[4]-1, el_ns[1]-1,
                        el_ns[5]-1, el_ns[2]-1, el_ns[6]-1,
                        el_ns[7]-1, el_ns[3]-1, el_ns[8]-1, 
                        el_ns[9]-1]
        vertices = np.array([(xp, yp, zp) for xp, yp, zp in af_def_coord[curr_nodes, :]])
        for edge in edges:
            x = [vertices[edge[0]][0], vertices[edge[1]][0]]
            y = [vertices[edge[0]][1], vertices[edge[1]][1]]
            z = [vertices[edge[0]][2], vertices[edge[1]][2]]
            ax.plot(x, y, z, c='k', alpha=0.5)

    # Axis 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('X - Strain')
    ax.scatter(coords[0, :], coords[1, :], coords[2, :], c=disp[0, :], cmap=cmap, s=100)
    cb = fig0.colorbar(ax.collections[0], ax=ax)
    cb.ax.tick_params(labelsize=5)

    plt.show()

def plot_laplace(u, np_nodes, disp, coords, np_eles):
    n_n    = int(len(np_nodes[:, 0]))

    # Node position matrices
    be_def_coord = np_nodes[:, 1:]
    shape_coords = np.zeros([n_n, 3])

    # Mapping node positions and displacements
    for z, _ in enumerate(np_nodes[:, 0]):
        shape_coords[z, 0] = be_def_coord[z, 0]
        shape_coords[z, 1] = be_def_coord[z, 1]
        shape_coords[z, 2] = be_def_coord[z, 2]

    # Set figures
    fig0 = plt.figure()

    # format
    cmap = get_cmap('Blues')
    ax = fig0.add_subplot(111, projection='3d')

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 6), (6, 7), (7, 9), (9, 2),
             (7, 8), (8, 4)]
    
    # Plot patches
    for _, el_ns in enumerate(np_eles):
        vertices = []
        # face = []
        curr_nodes = [el_ns[0]-1, el_ns[4]-1, el_ns[1]-1,
                        el_ns[5]-1, el_ns[2]-1, el_ns[6]-1,
                        el_ns[7]-1, el_ns[3]-1, el_ns[8]-1, 
                        el_ns[9]-1]
        vertices = np.array([(xp, yp, zp) for xp, yp, zp in shape_coords[curr_nodes, :]])
        for edge in edges:
            x = [vertices[edge[0]][0], vertices[edge[1]][0]]
            y = [vertices[edge[0]][1], vertices[edge[1]][1]]
            z = [vertices[edge[0]][2], vertices[edge[1]][2]]
            ax.plot(x, y, z, c='k', alpha=0.5)

    # Axis 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('X - Strain')
    ax.scatter(coords[0, :], coords[1, :], coords[2, :], c=disp.sum(axis=0), cmap=cmap, s=100)
    cb = fig0.colorbar(ax.collections[0], ax=ax)
    cb.ax.tick_params(labelsize=5)

    plt.show()

def plot_eps(u, coords, strains, np_nodes, np_eles):
    dim = 3
    n_n    = int(len(np_nodes[:, 0]))

    # Displacement component matrices
    u_x = np.zeros([n_n, 1])
    u_y = np.zeros([n_n, 1])
    u_z = np.zeros([n_n, 1])

    # Node position matrices
    be_def_coord = np_nodes[:, 1:]
    af_def_coord = np.zeros([n_n, 3])

    # Mapping node positions and displacements
    for z, _ in enumerate(np_nodes[:, 0]):
        u_x[z] = u[dim * z]
        u_y[z] = u[dim * z + 1]
        u_z[z] = u[dim * z + 2]
        af_def_coord[z, 0] = be_def_coord[z, 0] + u[dim * z]
        af_def_coord[z, 1] = be_def_coord[z, 1] + u[dim * z + 1]
        af_def_coord[z, 2] = be_def_coord[z, 2] + u[dim * z + 2]

    # Set figures
    fig = plt.figure()
    cmap = get_cmap('Blues')

    # x figure (stress, strain, displacement)
    ax0 = fig.add_subplot(111, projection='3d')

    # # Create triangulation
    # triang = mtri.Triangulation(coords[0, :], coords[1, :])

    # # strain x
    # tr = ax0.plot_trisurf(triang, strains[0, :], coords[2, :], cmap=cmap)
    # cb = fig.colorbar(tr, ax=ax0, orientation='vertical')
    # cb.ax.tick_params(labelsize=5)

    # Define the edges of the polyhedron
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 6), (6, 7), (7, 9), (9, 2),
             (7, 8), (8, 4)]

    if len(np_eles) == 10:
        vertices = []
        # face = []
        curr_nodes = [np_eles[0]-1, np_eles[4]-1, np_eles[1]-1,
                        np_eles[5]-1, np_eles[2]-1, np_eles[6]-1,
                        np_eles[7]-1, np_eles[3]-1, np_eles[8]-1, 
                        np_eles[9]-1]
        vertices = np.array([(xp, yp, zp) for xp, yp, zp in af_def_coord[curr_nodes, :]])
        for edge in edges:
            x = [vertices[edge[0]][0], vertices[edge[1]][0]]
            y = [vertices[edge[0]][1], vertices[edge[1]][1]]
            z = [vertices[edge[0]][2], vertices[edge[1]][2]]
            ax0.plot(x, y, z, c='b')

    else:
        # Plot patches
        for _, el_ns in enumerate(np_eles):
            vertices = []
            # face = []
            curr_nodes = [el_ns[0]-1, el_ns[4]-1, el_ns[1]-1,
                            el_ns[5]-1, el_ns[2]-1, el_ns[6]-1,
                            el_ns[7]-1, el_ns[3]-1, el_ns[8]-1, 
                            el_ns[9]-1]
            vertices = np.array([(xp, yp, zp) for xp, yp, zp in af_def_coord[curr_nodes, :]])
            for edge in edges:
                x = [vertices[edge[0]][0], vertices[edge[1]][0]]
                y = [vertices[edge[0]][1], vertices[edge[1]][1]]
                z = [vertices[edge[0]][2], vertices[edge[1]][2]]
                ax0.plot(x, y, z, c='b')

    # Format axis and plot scatters
    # ++++ X ++++
    # Axis 0
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Z')
    ax0.set_title('X - Stress')

    plt.show()

def nodes_and_elements(file_name, type_num):

    type_name = {2: "3-node-triangle", 3: "4-node-quadrangle", 4: "4-node-tetrahedron",  5: "8-node-hexahedron",  9: "6-node-second-order-triangle", 
                 10: "9-node-second-order-quadrangle",  11: "10-node-second-order-tetrahedron \n \n"}
    
    msh_file = open(file_name, 'r')

    # Find breaks which are indicated by '$'
    # Also retain information requires for Entities, Nodes and Elements
    # Set up containers and checks
    break_container = dict()
    nodes_list      = list()
    nod_check       = 0
    elements_list   = list()
    ele_check       = 0

    # Iterate
    for i, line in enumerate(msh_file):
        # Store break information
        if line[0][0] == '$':
            break_container[line[1:-1]] = i
        # Store information
        if nod_check:
            nodes_list.append(line[:-1])
        if ele_check:
            elements_list.append(line[:-1])
        # Checks on if we are in the right section
        if line[1:-1] == 'Nodes':
            nod_check = 1
            continue
        elif line[1:-1] == 'EndNodes':
            nod_check = 0
            continue
        elif line[1:-1] == 'Elements':
            ele_check = 1 
            continue
        elif line[1:-1] == 'EndElements':
            ele_check = 0
            continue

    # Remove the last value of each list to compensate for the title added
    nodes_list.pop()
    elements_list.pop()

    # Nodes 
    # Create a file to write to: gmsh2iron.node
    save_name = 'GitHub/Dino/' + file_name.split('gmsh_')[1].split('.msh')[0]
    node_file = open(save_name + '_cvt2dino.nodes', 'w')
    node_file.write(nodes_list[0] + "\n")
    # Loop through nodes and blocks
    node_positions = dict()
    count = 1
    for i, block in enumerate(nodes_list):
        if count:
            count -= 1
            continue
        # print(block.split(" ")[3])
        for j in range(int(block.split(" ")[3])):
            node_positions[int(nodes_list[i+j+1])] = (float(nodes_list[int(block.split(" ")[3])+i+j+1].split(" ")[0]), 
                                                      float(nodes_list[int(block.split(" ")[3])+i+j+1].split(" ")[1]), 
                                                      float(nodes_list[int(block.split(" ")[3])+i+j+1].split(" ")[2]))
            node_file.write(nodes_list[i+j+1]+"\t")
            node_file.write(nodes_list[int(block.split(" ")[3])+i+j+1].split(" ")[0] + "\t" +
                            nodes_list[int(block.split(" ")[3])+i+j+1].split(" ")[1] + "\t" +
                            nodes_list[int(block.split(" ")[3])+i+j+1].split(" ")[2] + "\n")
            count +=2

    # Elements
    # Create a file to store the element information: gmsh2iron.ele
    element_file = open(save_name + '_cvt2dino.ele', 'w')
    element_file.write(elements_list[0] + "\n")
    # Set Element Types
    types = {type_num: type_name[type_num].strip()}

    # Loop through nodes and blocks
    count           = 1
    for i, block in enumerate(elements_list):
        if count:
            count -= 1
            continue
        for j in range(int(block.split(" ")[3])):
            count +=1
            if int(block.split(" ")[2]) in types.keys():
                element_file.write(types[int(block.split(" ")[2])] + "\t")
                element_file.write(block.split(" ")[1] + "\t")
                for value in elements_list[i+j+1].split():
                    element_file.write(value + "\t")
                element_file.write("\n")
            else:
                continue

def deformation_gradient(X, x, we, points, sym_vars, delPhi, n_el_n=10, dim=3):

    F = np.zeros((3, 3))
    Fdef = np.zeros((3, 3))

    for q in range(0, dim, 1):
        displacement_gradient = np.zeros((3, 3))
        for i in range(n_el_n):
            displacement_gradient += np.outer(x[i] - X[i], delPhi[q][i])

        F += displacement_gradient + np.eye(3)

    for q, w in enumerate(we):
            # [1 2 3]
            Fdef[q+0, q+0] += w * float((F[0, 0]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            Fdef[q+0, q+1] += w * float((F[0, 1]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            Fdef[q+0, q+2] += w * float((F[0, 2]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            # [4 5 6]
            Fdef[q+1, q+0] += w * float((F[1, 0]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            Fdef[q+1, q+1] += w * float((F[1, 1]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            Fdef[q+1, q+2] += w * float((F[1, 2]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            # [4 5 6]
            Fdef[q+2, q+0] += w * float((F[2, 0]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            Fdef[q+2, q+1] += w * float((F[2, 1]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            Fdef[q+2, q+2] += w * float((F[2, 2]).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
        
    detF = np.linalg.det(F)
    print(Fdef)
    return Fdef, detF

    # J = np.gradient(x) / (np.gradient(X) + 1e-10)
    # Fdef = J + np.eye(3)
    # detF = np.linalg.det(Fdef)
    # return Fdef, detF

def ref_B_mat(e, np_n, np_e, x, delPhi, sym_vars, dim=3, n_el_n=10):
    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, 3)

    # Preallocate natural coordinates
    X_ele = np.zeros((n_el_n, dim))
    x_ele = np.zeros((n_el_n, dim))

    # Set rows and columns GLOBAL
    rc = np_e[e, :]

    # Find coordinates for those nodes
    np_n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        n_idx = np.where(np_n_idx == local_node)[0]
        X_ele[i] = np_n[n_idx, 1:dim+1][0]
        x_ele[i] = xc[n_idx, :][0]

    deformation_gradient(X_ele, x_ele, WE, GP, sym_vars, delPhi, n_el_n=10, dim=3)

    # Jacobian
    jac = delPhi * X_ele
    detJ = jac.det()
    b_sub = jac.inv() * delPhi

    # B Matrix
    b_mat = sym.zeros(6,dim*n_el_n)
    for r, c in enumerate(range(0, dim*n_el_n, dim)):
        Fdef, _ = deformation_gradient(X_ele[r], x_ele[r])
        # [F11φα,1 F21φα,1 F31φα,1] 
        b_mat[0, c] = Fdef[0,0] * b_sub[0, r]
        b_mat[0, c+1] = Fdef[1,0] * b_sub[0, r]
        b_mat[0, c+2] = Fdef[2,0] * b_sub[0, r]
        # [F12φα,2 F22φα,2 F32φα,2]
        b_mat[1, c] = Fdef[0,1] * b_sub[1, r]
        b_mat[1, c+1] = Fdef[1,1] * b_sub[1, r]
        b_mat[1, c+2] = Fdef[2,1] * b_sub[1, r]
        # [F13φα,3 F23φα,3 F33φα,3]
        b_mat[2, c] = Fdef[0,2] * b_sub[2, r]
        b_mat[2, c+1] = Fdef[1,2] * b_sub[2, r]
        b_mat[2, c+2] = Fdef[2,2] * b_sub[2, r]
        # [F11φα,2 + F12φα,1 F21φα,2 + F22φα,1 F31φα,2 + F32φα,1]
        b_mat[3, c] = Fdef[0,0] * b_sub[1, r] + Fdef[0,1] * b_sub[0, r]
        b_mat[3, c+1] = Fdef[1,0] * b_sub[1, r] + Fdef[1,1] * b_sub[0, r]
        b_mat[3, c+2] = Fdef[2,0] * b_sub[1, r] + Fdef[2,1] * b_sub[0, r]
        # [F12φα,3 + F13φα,2 F22φα,3 + F23φα,2 F32φα,3 + F33φα,2]
        b_mat[4, c] = Fdef[0,1] * b_sub[2, r] + Fdef[0,2] * b_sub[1, r]
        b_mat[4, c+1] = Fdef[1,1] * b_sub[2, r] + Fdef[1,2] * b_sub[1, r]
        b_mat[4, c+2] = Fdef[2,1] * b_sub[2, r] + Fdef[2,2] * b_sub[1, r]
        # [F13φα,1 + F11φα,3 F23φα,1 + F21φα,3 F33φα,1 + F31φα,3]  
        b_mat[5, c] = Fdef[0,2] * b_sub[0, r] + Fdef[0,0] * b_sub[2, r]
        b_mat[5, c+1] = Fdef[1,2] * b_sub[0, r] + Fdef[1,0] * b_sub[2, r]
        b_mat[5, c+2] = Fdef[2,2] * b_sub[0, r] + Fdef[2,0] * b_sub[2, r]

    return b_mat, detJ

def constitutive_eqs(e, con_type, c_vals, np_n, np_e, x, dim=3, n_el_n=10):
    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, 3)

    # Preallocate natural coordinates
    X_ele = np.zeros((n_el_n, dim))
    x_ele = np.zeros((n_el_n, dim))
    Smat = np.zeros((6, n_el_n))
    Dmat = np.zeros((6, n_el_n*6))

    # Set rows and columns GLOBAL
    rc = np_e[e, :]

    # Find coordinates for those nodes
    np_n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        n_idx = np.where(np_n_idx == local_node)[0]
        X_ele[i] = np_n[n_idx, 1:dim+1][0]
        x_ele[i] = xc[n_idx, :][0]

    for r in range(0, n_el_n, 1):
        Fdef, detF = deformation_gradient(X_ele[r], x_ele[r])
        Cgre = Fdef.T * Fdef
        # Mooney Rivlin
        # W = c1(I1 - 3) + c2(I2-3)
        if con_type == 0:
            # Derivatives of Energy in terms of Invariants
            # First Order
            dWdI = [c_vals[0], c_vals[1], 0]
            # Second Order
            ddWdII = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            sPK = second_piola(dWdI, Cgre, detF)
            dMo = elastic_moduli(dWdI, ddWdII, Cgre, detF)
        
        Smat[:, r] = sPK
        Dmat[:, 6*r:(6*r+6)] = dMo

    return Smat, Dmat

def second_piola(dWdI, C, Jc):
    sPK = np.zeros(6)
    diracD = np.eye(3)
    # Stress Indexes
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [0,2]])
    invC = np.linalg.inv(C)
    # Fill 2ndPK
    for i in range(0, 6, 1): 
        delIdelC = [diracD[ij[i,0], ij[i,1]],
                    np.trace(C) * diracD[ij[i,0], ij[i,1]] - C[ij[i,0], ij[i,1]],
                    0.5 * Jc * invC[ij[i,0], ij[i,1]]]
        sPk_i = np.matmul(delIdelC, np.array(dWdI))
        sPK[i] = 2 * sPk_i
    return np.transpose(sPK)

def elastic_moduli(dWdI, ddWdII, C, Jc):
    sPK = np.zeros(6)
    dd = np.eye(3)
    invC = np.linalg.inv(C)
    moduli = np.zeros((6,6))
    # Stress Indexes
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [0,2]])
    for r in range(0, 6, 1):
        term1 = 4 * np.matmul(
            [dd[ij[r,0], ij[r,1]],
             np.trace(C) * dd[ij[r,0], ij[r,1]] - C[ij[r,0], ij[r,1]],
             0.5 * Jc * invC[ij[r,0], ij[r,1]]
            ], ddWdII
        )
        term2 = [4*dWdI[0], dWdI[0]]
        for c in range(0, 6, 1):
            term3 = np.transpose(
                [dd[ij[c,0], ij[c,1]],
                 np.trace(C) * dd[ij[c,0], ij[c,1]] - C[ij[c,0], ij[c,1]],
                 0.5 * Jc * invC[ij[c,0], ij[c,1]]
                ]
            )
            lil_c = 0.5 * (invC[ij[r,0], ij[c,0]] * invC[ij[r,1], ij[c,1]] +
                           invC[ij[r,0], ij[c,1]] * invC[ij[r,1], ij[c,0]])
            term4 = [dd[ij[r,0], ij[r,1]] * dd[ij[c,0], ij[c,1]] - 
                     0.5 * (dd[ij[r,0], ij[c,0]] * dd[ij[r,1], ij[c,1]] +
                             dd[ij[r,0], ij[c,1]] * dd[ij[r,1], ij[c,0]]),
                     Jc * (invC[ij[r,0], ij[r,1]] * invC[ij[c,0], ij[c,1]] -
                           2*lil_c)
            ]
            moduli[r, c] = np.matmul(term1, term3) + np.matmul(term4, term2)
    return moduli

def geometric_tangent_k(e, Spk, delPhi, np_e, k_geo, dim=3, n_el_n=10):
    Spk = Spk[e]
    rc = np_e[e, :]
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [0,2]])
    for r, n in enumerate(rc):
        Idn = sym.eye(3)
        Geo = sym.zeros(n_el_n, n_el_n)
        Kgeo = sym.zeros(3, 3)
        for a in range(0, n_el_n, 1):
            for b in range(0, n_el_n, 1):
                Geo[a, b] = delPhi[ij[0,0], a] * Spk[ij[0,0], ij[0,1]] * delPhi[ij[0,1], b] + \
                            delPhi[ij[1,0], a] * float(Spk[ij[1,0], ij[1,1]]) * delPhi[ij[1,1], b] + \
                            delPhi[ij[2,0], a] * float(Spk[ij[2,0], ij[1,1]]) * delPhi[ij[2,1], b] + \
                            delPhi[ij[3,0], a] * float(Spk[ij[3,0], ij[3,1]]) * delPhi[ij[3,1], b] + \
                            delPhi[ij[4,0], a] * float(Spk[ij[4,0], ij[4,1]]) * delPhi[ij[4,1], b] + \
                            delPhi[ij[5,0], a] * float(Spk[ij[5,0], ij[5,1]]) * delPhi[ij[5,1], b]
                Kgeo = Kgeo + Geo[a, b] * Idn
        k_geo[dim*(n-1):dim*(n-1)+3, dim*(n-1):dim*(n-1)+3] = k_geo[dim*(n-1):dim*(n-1)+3, dim*(n-1):dim*(n-1)+3] + Kgeo
    return k_geo
    
def gauss_int_fsol(e, weights, points, Bmat, detJ, Smat, sym_vars, k_sol, np_e, n_el_n=10, dim=3):
    Bmat = Bmat[e]
    BTra = np.transpose(Bmat)
    Smat = Smat[e]
    detJ = detJ[e]
    rc = np_e[e, :]
    term_sol = np.zeros(Bmat.shape[1])
    # Gauss Quadrature
    for r, n in enumerate(rc):
        i = r*3
        term = np.matmul(BTra[i:i+3, :], Smat[:, r])
        i = r
        for q, w in enumerate(weights):
            # [1 2 3]
            term_sol[dim*i+0] = term_sol[dim*i+0] + w * \
                                 float((term[0]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            term_sol[dim*i+1] = term_sol[dim*i+1] + w * \
                                 float((term[1]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            term_sol[dim*i+2] = term_sol[dim*i+2] + w * \
                                 float((term[2]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))

        k_sol[dim*(n-1)+0] = k_sol[dim*(n-1)+0] + term_sol[dim*r+0] 
        k_sol[dim*(n-1)+1] = k_sol[dim*(n-1)+1] + term_sol[dim*r+1] 
        k_sol[dim*(n-1)+2] = k_sol[dim*(n-1)+2] + term_sol[dim*r+2] 

    return k_sol

def gauss_int_ftan(e, weights, points, Bmat, detJ, Dmat, Gmat, sym_vars, k_tan, np_e, n_el_n=10, dim=3):
    Bmat = Bmat[e]
    BTra = np.transpose(Bmat)
    Dmat = Dmat[e]
    Gmat = Gmat[e]
    detJ = detJ[e]
    rc = np_e[e, :]
    term_tan = np.zeros((Bmat.shape[1], Bmat.shape[1]))
    # Gauss Quadrature
    for r, n in enumerate(rc):
        i = r*3
        term = np.matmul(BTra[i:i+3, :], Dmat[:, i:i+6])
        term = np.matmul(term, Bmat[:, i:i+3]) + Gmat[dim*(n-1):dim*(n-1)+3, dim*(n-1):dim*(n-1)+3]
        i = r
        for q, w in enumerate(weights):
            # [1 2 3]
            term_tan[dim*i+0, dim*i+0] = term_tan[dim*i+0, dim*i+0] + w * \
                                          float((term[0, 0]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            term_tan[dim*i+0, dim*i+1] = term_tan[dim*i+0, dim*i+1] + w * \
                                          float((term[0, 1]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            term_tan[dim*i+0, dim*i+2] = term_tan[dim*i+0, dim*i+2] + w * \
                                          float((term[0, 2]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            # [4 5 6]
            term_tan[dim*i+1, dim*i+0] = term_tan[dim*i+1, dim*i+0] + w * \
                                          float((term[1, 0]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            term_tan[dim*i+1, dim*i+1] = term_tan[dim*i+1, dim*i+1] + w * \
                                          float((term[1, 1]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            term_tan[dim*i+1, dim*i+2] = term_tan[dim*i+1, dim*i+2] + w * \
                                          float((term[1, 2]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            # [4 5 6]
            term_tan[dim*i+2, dim*i+0] = term_tan[dim*i+2, dim*i+0] + w * \
                                          float((term[2, 0]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            term_tan[dim*i+2, dim*i+1] = term_tan[dim*i+2, dim*i+1] + w * \
                                          float((term[2, 1]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))
            term_tan[dim*i+2, dim*i+2] = term_tan[dim*i+2, dim*i+2] + w * \
                                          float((term[2, 2]*detJ).subs({sym_vars[0]: points[q, 0], sym_vars[1]: points[q, 1], sym_vars[2]: points[q, 2]}))

        # [1 2 3]
        k_tan[dim*(n-1)+0, dim*(n-1)+0] = k_tan[dim*(n-1)+0, dim*(n-1)+0] + term_tan[dim*r+0, dim*r+0] 
        k_tan[dim*(n-1)+0, dim*(n-1)+1] = k_tan[dim*(n-1)+0, dim*(n-1)+1] + term_tan[dim*r+0, dim*r+1] 
        k_tan[dim*(n-1)+0, dim*(n-1)+2] = k_tan[dim*(n-1)+0, dim*(n-1)+2] + term_tan[dim*r+0, dim*r+2] 
        # [4 5 6]
        k_tan[dim*(n-1)+1, dim*(n-1)+0] = k_tan[dim*(n-1)+1, dim*(n-1)+0] + term_tan[dim*r+1, dim*r+0] 
        k_tan[dim*(n-1)+1, dim*(n-1)+1] = k_tan[dim*(n-1)+1, dim*(n-1)+1] + term_tan[dim*r+1, dim*r+1] 
        k_tan[dim*(n-1)+1, dim*(n-1)+2] = k_tan[dim*(n-1)+1, dim*(n-1)+2] + term_tan[dim*r+1, dim*r+2] 
        # [7 8 9]
        k_tan[dim*(n-1)+2, dim*(n-1)+0] = k_tan[dim*(n-1)+2, dim*(n-1)+0] + term_tan[dim*r+2, dim*r+0] 
        k_tan[dim*(n-1)+2, dim*(n-1)+1] = k_tan[dim*(n-1)+2, dim*(n-1)+1] + term_tan[dim*r+2, dim*r+1] 
        k_tan[dim*(n-1)+2, dim*(n-1)+2] = k_tan[dim*(n-1)+2, dim*(n-1)+2] + term_tan[dim*r+2, dim*r+2] 

    return k_tan

def calc_nonlinear_func(u, np_n, np_e, w, p, delPhi, sym_vars, con_type, c_vals, dim, n_el_n, n_ele, num_pro):
    n_n = int(len(np_n[:, 0]))
    x = np_n[:, 1:].flatten() 
    x = x + u

    k_sol = np.zeros(dim*n_n)

    Bmat_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Smat_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Fsol_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)

    ele_list = range(0, n_ele, 1)

    p_Bmat = partial(ref_B_mat, np_n=np_n, np_e=np_e, x=x, delPhi=delPhi, sym_vars=sym_vars, dim=dim, n_el_n=n_el_n)
    p_Smat = partial(constitutive_eqs, \
                     con_type=con_type, c_vals=c_vals, np_n=np_n, np_e=np_e, x=x, dim=dim, \
                        n_el_n=n_el_n)

    Bmat_Results = Bmat_Pool.map(p_Bmat, ele_list)
    Bmats, detJs = zip(*Bmat_Results)
    Smat_Results = Smat_Pool.map(p_Smat, ele_list)
    Smats, _ = zip(*Smat_Results)

    p_Fsol = partial(gauss_int_fsol, \
                     weights=w, points=p, Bmat=Bmats, detJ=detJs, Smat=Smats, sym_vars=sym_vars, \
                        k_sol=k_sol, np_e=np_e, n_el_n=n_el_n, dim=dim)
    Fsol_Results = Fsol_Pool.map(p_Fsol, ele_list)

    nrFunc = sum(Fsol_Results)

    return nrFunc

def calc_nonlinear_tang(u, np_n, np_e, w, p, delPhi, sym_vars, con_type, c_vals, dim, n_el_n, n_ele, num_pro):
    n_n = int(len(np_n[:, 0]))
    x = np_n[:, 1:].flatten() 
    x = x + u

    k_tan = np.zeros((dim*n_n, dim*n_n))
    k_geo = sym.zeros(dim*n_n, dim*n_n)

    Bmat_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Smat_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Gmat_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Ftan_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)

    ele_list = range(0, n_ele, 1)

    p_Bmat = partial(ref_B_mat, np_n=np_n, np_e=np_e, x=x, delPhi=delPhi, sym_vars=sym_vars, dim=dim, n_el_n=n_el_n)
    p_Smat = partial(constitutive_eqs, \
                      con_type=con_type, c_vals=c_vals, np_n=np_n, np_e=np_e, x=x, dim=dim, \
                       n_el_n=n_el_n)

    Bmat_Results = Bmat_Pool.map(p_Bmat, ele_list)
    Bmats, detJs = zip(*Bmat_Results)
    Smat_Results = Smat_Pool.map(p_Smat, ele_list)
    Smats, Dmats = zip(*Smat_Results)

    p_Gmat = partial(geometric_tangent_k, Spk=Smats, delPhi=delPhi, np_e=np_e, k_geo=k_geo, dim=dim, n_el_n=n_el_n)
    Kgeos = Gmat_Pool.map(p_Gmat, ele_list)

    p_Fsol = partial(gauss_int_ftan, \
                     weights=w, points=p, Bmat=Bmats, detJ=detJs, Dmat=Dmats, Gmat=Kgeos, sym_vars=sym_vars, \
                        k_tan=k_tan, np_e=np_e, n_el_n=n_el_n, dim=dim)
    Ftan_Results = Ftan_Pool.map(p_Fsol, ele_list)

    nrFtan = sum(Ftan_Results)

    return nrFtan

def apply_nonlinear_BC(np_n, u, BC_0, BC_1, axi, dim=3):

    nodes = list()

    min_val = np.amin(np_n[:, axi+1])
    max_val = np.amax(np_n[:, axi+1])

    # Apply Bounadry Conditions
    # Check position of node and apply BC if condition met
    for n in np_n[:, 0]:
        n_idx = int(n)
        n_val = np_n[np_n[:, 0] == n, axi+1][0]

        if n_val == min_val:
            u[dim*(n_idx-1)+axi] = BC_0
            nodes.append(dim*(n_idx-1)+axi)

        elif n_val == max_val:
            u[dim*(n_idx-1)+axi] = BC_1
            nodes.append(dim*(n_idx-1)+axi)

    return u, nodes

def newton_raph(u, iter, tol, nodes, np_n, np_e, w, p, delPhi, sym_vars, con_type, c_vals, dim, n_el_n, n_ele, num_pro):
    for i in range(0, iter, 1):
        nrFunc = calc_nonlinear_func(u, np_n, np_e, w, p, delPhi, sym_vars, con_type, c_vals, dim, n_el_n, n_ele, num_pro)
        print(nrFunc)
        nrFtan = calc_nonlinear_tang(u, np_n, np_e, w, p, delPhi, sym_vars, con_type, c_vals, dim, n_el_n, n_ele, num_pro)
        print(nrFtan)
        newtStep = np.matmul(np.linalg.inv(nrFtan), nrFunc)
        newtStep[nodes] = 0
        da = u - newtStep
        diff = np.average(da - u)
        print("Iteration Number: {}".format(i))
        print("Average Difference: {}".format(diff))
        if diff < tol:
            return da, i
        u = da
    
    print("Did not converge")
    return u, iter

# u         = np.loadtxt("output_files/u_" + FILE_NAME + ".text", dtype=float)
# k_gl      = np.loadtxt("output_files/k_" + FILE_NAME + ".text", dtype=float)
# disp      = np.loadtxt("output_files/disp_" + FILE_NAME + ".text", dtype=float)
# coords    = np.loadtxt("output_files/coords_" + FILE_NAME + ".text", dtype=float)
# undef     = np.loadtxt("output_files/undefc_" + FILE_NAME + ".text", dtype=float)
# strains   = np.loadtxt("output_files/strain_" + FILE_NAME + ".text", dtype=float)
# np_nodes  = np.loadtxt("output_files/np_nodes_" + FILE_NAME + ".text", dtype=float)
# np_eles   = np.loadtxt("output_files/np_eles_" + FILE_NAME + ".text", dtype=int)

# plot_disps(u, np_nodes, strains, coords, np_eles)

# plot_laplace(u, np_nodes, strains, undef, np_eles)

# plot_eps(u, coords, strains, np_nodes, np_eles)

# post_calc(k_gl, np_nodes, np_eles, phi, delPhi)

# nodes_and_elements("GitHub/Dino/gmsh_cubeTest.msh", 11)
# dim, n_el_n, sym_vars, phi, delPhi = element_assign(el_type=1, el_order=1)
# nodes = open("GitHub/Dino/cubeTest_cvt2dino.nodes", 'r')
# elems = open("GitHub/Dino/cubeTest_cvt2dino.ele", 'r')

# # Store Node and Element Data
# nodes_list = list()
# elements_list = list()
# for line in nodes:
#     nodes_list.append(line.strip().replace('\t', ' ').split(' '))
# for line in elems:
#     elements_list.append(line.strip().replace('\t', ' ').split(' '))
# np_nodes  = np.array(nodes_list[1:])
# np_nodes  = np_nodes.astype(float)
# np_eles   = np.array(elements_list[1:])
# np_eles   = np_eles[:, 3:].astype(int)

# # Determine Parameters
# n_ele = len(np_eles[:, 0])
# n_n = int(len(np_nodes[:, 0]))

# coords = np.zeros((n_el_n, dim))
# rc = np_eles[0, :]
# np_nodes_idx = np_nodes[:, 0]
# for i, local_node in enumerate(rc):
#     node_indices = np.where(np_nodes_idx == local_node)[0]
#     coords[i] = np_nodes[node_indices, 1:dim+1][0]

# Fdef, Cgre = deformation_gradient(coords[0], coords[0])

# Cgre = np.array([[1,0,0],[0,1,0],[0,0,1]])

# sPK, dMo = constitutive_eqs(0, [2, 3, 50], Cgre)

# print(sPK)
# print(dMo)