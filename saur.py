import numpy as np
import sympy as sym
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from functools import partial

DIM = 3
N_EL_N = 10
ORDER = 5
I = np.eye(DIM)
WE = np.array([-4/5, 9/20, 9/20, 9/20, 9/20])
GP = np.array(
    [
        [1/4, 1/4, 1/4], 
        [1/2, 1/6, 1/6], 
        [1/6, 1/2, 1/6], 
        [1/6, 1/6, 1/2],
        [1/6, 1/6, 1/6]
    ]
)
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
# dNdxez = [δφ1/δξ δφ2/δξ ... δφ10/δξ
#           δφ1/δξ δφ2/δξ ... δφ10/δξ
#           δφ1/δζ δφ2/δζ ... δφ10/δζ]
dNdxez = sym.Matrix(
    [
        [sym.diff(phi[j], xi, 1) for j in range(0, N_EL_N, 1)],  
        [sym.diff(phi[k], eta, 1) for k in range(0, N_EL_N, 1)],
        [sym.diff(phi[m], zeta, 1) for m in range(0, N_EL_N, 1)]
    ]
)

def apply_nonlinear_BC(np_n, u, nodes, BC0, BC1, axi):

    min_val = np.amin(np_n[:, axi+1])
    max_val = np.amax(np_n[:, axi+1])

    # Apply Bounadry Conditions
    # Check position of node and apply BC if condition met
    for n in np_n[:, 0]:
        n_idx = int(n)
        n_val = np_n[np_n[:, 0] == n, axi+1][0]

        if n_val == min_val:
            if BC0[0] is not None:
                u[DIM*(n_idx-1)+0] = BC0[0]
                nodes.append(DIM*(n_idx-1)+0)
            if BC0[1] is not None:
                u[DIM*(n_idx-1)+1] = BC0[1]
                nodes.append(DIM*(n_idx-1)+1)
            if BC0[2] is not None:
                u[DIM*(n_idx-1)+2] = BC0[2]
                nodes.append(DIM*(n_idx-1)+2)

        elif n_val == max_val:
            if BC1[0] is not None:
                u[DIM*(n_idx-1)+0] = BC1[0]
                nodes.append(DIM*(n_idx-1)+0)
            if BC1[1] is not None:
                u[DIM*(n_idx-1)+1] = BC1[1]
                nodes.append(DIM*(n_idx-1)+1)
            if BC1[2] is not None:
                u[DIM*(n_idx-1)+2] = BC1[2]
                nodes.append(DIM*(n_idx-1)+2)

    return u, nodes

def cur_B_mat(e, np_n, np_e, x):

    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)
    rc = np_e[e, :]

    # Preallocate natural coordinates
    x_ele = np.zeros((N_EL_N, DIM))

    # Find coordinates for those nodes
    np_n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        n_idx = np.where(np_n_idx == local_node)[0]
        x_ele[i] = xc[n_idx, :][0]

    # Jacobian
    jac = dNdxez * x_ele
    detJ  = jac.det()
    dNdxyz = jac.inv() * dNdxez

    # Initialize a matrix filled with zeros
    b_mat = sym.zeros(DIM*2, N_EL_N*DIM)

    # Loop through each column index c
    for r, c in enumerate(range(0, DIM*N_EL_N, DIM)):
        # [δφ/δx,      0,      0] 
        b_mat[0, c+0] = dNdxyz[0, r]
        # [    0,  δφ/δy,      0]
        b_mat[1, c+1] = dNdxyz[1, r]
        # [    0,      0,  δφ/δz]
        b_mat[2, c+2] = dNdxyz[2, r]
        # [δφ/δy,  δφ/δx,      0]
        b_mat[3, c+0] = dNdxyz[1, r]
        b_mat[3, c+1] = dNdxyz[0, r]
        # [    0,  δφ/δz,  δφ/δy]
        b_mat[4, c+1] = dNdxyz[2, r]
        b_mat[4, c+2] = dNdxyz[1, r]
        # [δφ/δz,      0,  δφ/δx]  
        b_mat[5, c+0] = dNdxyz[2, r]
        b_mat[5, c+2] = dNdxyz[0, r] 

    return b_mat, detJ

def constitutive_eqs(e, c_vals, np_n, np_e, x):

    d = 1000.10

    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)
    
    # Preallocate natural coordinates
    X_ele = np.zeros((N_EL_N, DIM))
    x_ele = np.zeros((N_EL_N, DIM))
    sig = sym.zeros(DIM*2, 1)
    Dmat = sym.zeros(DIM*2, DIM*2)

    # Set rows and columns GLOBAL
    rc = np_e[e, :]

    # Find coordinates for those nodes
    np_n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        n_idx = np.where(np_n_idx == local_node)[0]
        X_ele[i] = np_n[n_idx, 1:DIM+1][0]
        x_ele[i] = xc[n_idx, :][0]

    cur = sym.Matrix(x_ele)
    ref = sym.Matrix(X_ele)

    # Determine Jacobians 
    jcur = dNdxez * cur
    jref = dNdxez * ref
    invJref = jref.inv()
    symF = invJref * jcur
    detF = symF.det()

    b = symF * symF.T
    trb = sym.trace(b)
    # Mooney Rivlin
    # W = c1(I1 - 3) + c2(I2-3) + d*(J-1)^2
    # Derivatives of Energy in terms of Invariants
    # First Order
    dWdI = sym.Matrix([c_vals[0], c_vals[1], 2*d*(detF-1)])
    # Second 
    ddWdII = sym.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 2*d]])
    sig = cauchy(dWdI, b, trb, detF)
    Dmat = d_moduli(dWdI, ddWdII,  b, trb, detF)
    
    return sig, Dmat

def cauchy(dWdI, b, trb, jac):
    s = sym.zeros(DIM, DIM)
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
    for _, (i, j) in enumerate(ij):
        dIdb = sym.Matrix(
            [
                [
                    b[i, j], 
                    trb * b[i, j] - (b[i, 0]*b[0, j] + b[i, 1]*b[1, j] + b[i, 2]*b[2, j]), 
                    0.5 * jac * I[i, j]
                ]
            ]
        )
        sig = dIdb * dWdI
        s[i, j] = 2 * sig / jac
    sig = sym.Matrix([[s[0,0], s[1,1], s[2,2], s[0,1], s[1,2], s[2,0]]])
    return sig

def d_moduli(dWdI, ddWdII,  b, trb, jac):
    d = sym.zeros(DIM*2,DIM*2)
    # Stress Indexes
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
    for r, (i, j) in enumerate(ij):
        term1 = sym.Matrix(
            [
                [
                    b[i, j], 
                    trb * b[i, j] - (b[i, 0]*b[0, j] + b[i, 1]*b[1, j] + b[i, 2]*b[2, j]), 
                    0.5 * jac * I[i, j]
                ]
            ]
        )
        term1 = 4 * (term1 * ddWdII)
        term4 = sym.Matrix(
            [
                [4*dWdI[0]], 
                [dWdI[0]]
            ]
        )
        for c, (k, l) in enumerate(ij):
            term2 = sym.Matrix(
                [
                    [b[k, l]],
                    [trb * b[k, l] - (b[k, 0]*b[0, l] + b[k, 1]*b[1, l] + b[k, 2]*b[2, l])],
                    [0.5 * jac * I[k, l]]
                ]
            )
            lil_b = 0.5 * (I[i, k] * I[j, l] + I[i, l] * I[j, k])
            term3 = sym.Matrix(
                [
                    [
                        b[i, j] *  b[k, l] - 0.5 * (b[i, k] * b[j, l] + b[i, l] * b[j, k]),
                        jac * (I[i, j] * I[k, l] - 2*lil_b) ################
                    ]
                ]
            )
            d[r, c] = (term1 * term2) + (term3 * term4)
    return d

def geometric_k(e, x, sig, np_n, np_e, k_geo, detJ):

    s = sig[e]
    jac = detJ[e]
    rc = np_e[e, :]
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])

    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)

    # Preallocate natural coordinates
    x_ele = np.zeros((N_EL_N, DIM))

    # Find coordinates for those nodes
    np_n_idx = np_n[:, 0]

    for i, local_node in enumerate(rc):
        n_idx = np.where(np_n_idx == local_node)[0]
        x_ele[i] = xc[n_idx, :][0]

    # Jacobian
    jcur = dNdxez * x_ele
    dNdxyz = jcur.inv() * dNdxez

    for a in range(0, N_EL_N, 1):
        for b in range(0, N_EL_N, 1):
            Gab = 0
            g = 0
            for n, (i, j) in enumerate(ij):
                Gab += dNdxyz[i, a] * s[0, n] * dNdxyz[j, b]
            for q, w in enumerate(WE):
                g += w * float((Gab * jac).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
            k_geo[DIM*(rc[a]-1):DIM*(rc[a]-1)+3, DIM*(rc[b]-1):DIM*(rc[b]-1)+3] += g * I

    return k_geo
    
def gauss_int_fsol(e, Bmat, detJ, sig, k_sol, np_e):

    bT = np.transpose(Bmat[e])
    s = sig[e]
    jac = detJ[e]
    rc = np_e[e, :]
    BTsig = np.zeros(DIM*N_EL_N)
    bTs = -1 * (jac * (bT * s.T))

    for i in range(0, N_EL_N, 1):
        for q, w in enumerate(WE):
            BTsig[DIM*i+0] += w * float((bTs[DIM*i+0]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
            BTsig[DIM*i+1] += w * float((bTs[DIM*i+1]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
            BTsig[DIM*i+2] += w * float((bTs[DIM*i+2]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
            

        k_sol[DIM*(rc[i]-1):DIM*(rc[i]-1)+3] += BTsig[DIM*i:DIM*i+3]

    return k_sol

def gauss_int_ftan(e, Bmat, detJ, Dmat, k_tan, np_e):

    b = Bmat[e]
    bT = np.transpose(b)
    d = Dmat[e]
    jac = detJ[e]

    rc = np_e[e, :]
    BTdB = np.zeros((DIM*N_EL_N, DIM*N_EL_N))
    bTdb = np.matmul(np.matmul(bT, d), b) * jac

    # Gauss Quadrature
    for i in range(0, N_EL_N, 1):
        for j in range(0, N_EL_N, 1):
            for q, w in enumerate(WE):
                # [1, 2, 3]
                BTdB[DIM*i+0, DIM*j+0] += w * float((bTdb[DIM*i+0, DIM*j+0]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
                BTdB[DIM*i+0, DIM*j+1] += w * float((bTdb[DIM*i+0, DIM*j+1]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
                BTdB[DIM*i+0, DIM*j+2] += w * float((bTdb[DIM*i+0, DIM*j+2]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
                # [4, 5, 3]
                BTdB[DIM*i+1, DIM*j+0] += w * float((bTdb[DIM*i+1, DIM*j+0]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
                BTdB[DIM*i+1, DIM*j+1] += w * float((bTdb[DIM*i+1, DIM*j+1]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
                BTdB[DIM*i+1, DIM*j+2] += w * float((bTdb[DIM*i+1, DIM*j+2]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
                # [1, 2, 3]
                BTdB[DIM*i+2, DIM*j+0] += w * float((bTdb[DIM*i+2, DIM*j+0]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
                BTdB[DIM*i+2, DIM*j+1] += w * float((bTdb[DIM*i+2, DIM*j+1]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))
                BTdB[DIM*i+2, DIM*j+2] += w * float((bTdb[DIM*i+2, DIM*j+2]).subs({xi: GP[q, 0], eta: GP[q, 1], zeta: GP[q, 2]}))

            k_tan[DIM*(rc[i]-1):DIM*(rc[i]-1)+3, DIM*(rc[j]-1):DIM*(rc[j]-1)+3] += BTdB[DIM*i:DIM*i+3, DIM*j:DIM*j+3]

    return k_tan

def nonlinear_solve(u, np_n, np_e, c_vals, n_ele, num_pro):

    n_n = int(len(np_n[:, 0]))
    x = np_n[:, 1:].flatten() 
    x = x + u

    k_sol = np.zeros(DIM*n_n)
    k_tan = np.zeros((DIM*n_n, DIM*n_n))
    k_geo = np.zeros((DIM*n_n, DIM*n_n))

    Bmat_Pool = mp.Pool(processes=num_pro)
    sig_Pool = mp.Pool(processes=num_pro)
    Gmat_Pool = mp.Pool(processes=num_pro)
    Fsol_Pool = mp.Pool(processes=num_pro)
    Ftan_Pool = mp.Pool(processes=num_pro)

    ele_list = range(0, n_ele, 1)

    p_Bmat = partial(cur_B_mat, np_n=np_n, np_e=np_e, x=x)
    p_sig = partial(constitutive_eqs, c_vals=c_vals, np_n=np_n, np_e=np_e, x=x)

    Bmat_Results = Bmat_Pool.map(p_Bmat, ele_list)
    Bmats, detJs = zip(*Bmat_Results)
    sig_Results = sig_Pool.map(p_sig, ele_list)
    sigs, Dmats = zip(*sig_Results)

    p_Gmat = partial(geometric_k, x=x, sig=sigs, np_n=np_n, np_e=np_e, k_geo=k_geo, detJ=detJs)
    Kgeos = Gmat_Pool.map(p_Gmat, ele_list)
    Gmat = sum(Kgeos)

    p_Fsol = partial(gauss_int_fsol, Bmat=Bmats, detJ=detJs, sig=sigs, k_sol=k_sol, np_e=np_e)
    p_Ftan = partial(gauss_int_ftan, Bmat=Bmats, detJ=detJs, Dmat=Dmats,  k_tan=k_tan, np_e=np_e)
    
    Fsol_Results = Fsol_Pool.map(p_Fsol, ele_list)
    nrFunc = sum(Fsol_Results)
    Ftan_Results = Ftan_Pool.map(p_Ftan, ele_list)
    nrFtan = sum(Ftan_Results)

    return nrFunc, (nrFtan + Gmat)

def newton_raph(xn, nodes, np_n, np_e, n_ele, c_vals, num_pro, iters, tol):
    for i in range(0, iters, 1):
        nrFunc, nrFtan = nonlinear_solve(xn, np_n, np_e, c_vals, n_ele, num_pro)
        nrFtanSol = np.copy(nrFtan)
        if nodes != None:
            nrFunc[nodes] = 0
            for idx in nodes:
                nrFtanSol[idx, :] = 0
                nrFtanSol[:, idx] = 0
                nrFtanSol[idx, idx] = nrFtan[idx, idx]
        newtStep = np.matmul(np.linalg.inv(nrFtanSol), nrFunc)
        xn1 = xn - newtStep
        xn1 = np.round(xn1, 10)
        xn = xn1

        print("Residual Average: {}".format(np.average(nrFunc)))
        print("Iteration Number: {}".format(i))

        if abs(np.average(nrFunc)) < tol:
            return xn1, i

    print("Did not converge")
    return xn, iters

def plot_disps(np_n, np_e, u, n_ele):
    plt.plot(u)
    plt.show()
    cmap = get_cmap('seismic')
    # cmap = get_cmap('Blues')

    xez, phi, _ = element_assign()
    _, GP = gauss_num_int()
    x_gp = np.zeros(n_ele*len(GP))
    y_gp = np.zeros(n_ele*len(GP))
    z_gp = np.zeros(n_ele*len(GP))
    u_gp = np.zeros(n_ele*len(GP))
    v_gp = np.zeros(n_ele*len(GP))
    w_gp = np.zeros(n_ele*len(GP))

    # Store disp vals
    uvw = sym.zeros(N_EL_N, DIM)
    xyz = sym.zeros(N_EL_N, DIM)
                       
    for e in range(0, n_ele, 1):
        # Set rows and columns GLOBAL
        rc = np_e[e, :]

        # Find coordinates and displacements
        for i, n in enumerate(rc):
            uvw[i, 0] = u[(rc[i]-1)*DIM]
            uvw[i, 1] = u[(rc[i]-1)*DIM+1]
            uvw[i, 2] = u[(rc[i]-1)*DIM+2]
            xyz[i, 0] = np_n[np.where(np_n[:, 0] == n), 1][0]
            xyz[i, 1] = np_n[np.where(np_n[:, 0] == n), 2][0]
            xyz[i, 2] = np_n[np.where(np_n[:, 0] == n), 3][0]
        
        # Set x and y, u and v, in terms of shape functions
        x    = phi.T * xyz[:, 0]
        y    = phi.T * xyz[:, 1]
        z    = phi.T * xyz[:, 2]
        u_sf = phi.T * uvw[:, 0]
        v_sf = phi.T * uvw[:, 1]
        w_sf = phi.T * uvw[:, 2]

        for q, p in enumerate(GP):
            x_gp[e*len(GP)-1 + q] = float((u_sf[0]+x[0]).subs({xi: p[0], eta: p[1], zeta: p[2]}))
            y_gp[e*len(GP)-1 + q] = float((v_sf[0]+y[0]).subs({xi: p[0], eta: p[1], zeta: p[2]}))
            z_gp[e*len(GP)-1 + q] = float((w_sf[0]+z[0]).subs({xi: p[0], eta: p[1], zeta: p[2]}))
            u_gp[e*len(GP)-1 + q] = float((u_sf[0]).subs({xi: p[0], eta: p[1], zeta: p[2]}))
            v_gp[e*len(GP)-1 + q] = float((v_sf[0]).subs({xi: p[0], eta: p[1], zeta: p[2]}))
            w_gp[e*len(GP)-1 + q] = float((w_sf[0]).subs({xi: p[0], eta: p[1], zeta: p[2]}))
    
    Ugp = np.array([u_gp, v_gp, w_gp]) 
    Cgp = np.array([x_gp, y_gp, z_gp]) 

    n_n    = int(len(np_n[:, 0]))

    # Node position matrices
    refX = np_n[:, 1:]
    curx = (refX.flatten() + u).reshape(n_n, 3)

    # Set figures
    fig0 = plt.figure()

    ax0 = fig0.add_subplot(221, projection='3d')
    ax1 = fig0.add_subplot(222, projection='3d')
    ax2 = fig0.add_subplot(223, projection='3d')
    ax3 = fig0.add_subplot(224, projection='3d')

    egs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 6), (6, 7), (7, 9), (9, 2),
             (7, 8), (8, 4)]
    
    # Plot patches
    for _, el_ns in enumerate(np_e):
        vtsx = []
        vtsX = []
        cNode = [el_ns[0]-1, el_ns[4]-1, el_ns[1]-1,
                        el_ns[5]-1, el_ns[2]-1, el_ns[6]-1,
                        el_ns[7]-1, el_ns[3]-1, el_ns[8]-1, 
                        el_ns[9]-1]
        vtsx = np.array([(xp, yp, zp) for xp, yp, zp in curx[cNode, :]])
        vtsX = np.array([(Xp, Yp, Zp) for Xp, Yp, Zp in refX[cNode, :]])
        for edge in egs:
            x = [vtsx[edge[0]][0], vtsx[edge[1]][0]]
            y = [vtsx[edge[0]][1], vtsx[edge[1]][1]]
            z = [vtsx[edge[0]][2], vtsx[edge[1]][2]]
            X = [vtsX[edge[0]][0], vtsX[edge[1]][0]]
            Y = [vtsX[edge[0]][1], vtsX[edge[1]][1]]
            Z = [vtsX[edge[0]][2], vtsX[edge[1]][2]]
            ax0.plot(X, Y, Z, c='k', alpha=1)
            ax1.plot(x, y, z, c='k', alpha=0.5)
            ax2.plot(x, y, z, c='k', alpha=0.5)
            ax3.plot(x, y, z, c='k', alpha=0.5)

    # Axis 
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Z')
    ax0.set_title('Reference')
    ax0.tick_params(labelsize=5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Current X-Disp')
    ax1.tick_params(labelsize=5)
    ax1.scatter(Cgp[0, :], Cgp[1, :], Cgp[2, :], c=Ugp[0, :], cmap=cmap, s=100)
    cb = fig0.colorbar(ax1.collections[0], ax=ax1)
    cb.ax.tick_params(labelsize=5)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Current Y-Disp')
    ax2.tick_params(labelsize=5)
    ax2.scatter(Cgp[0, :], Cgp[1, :], Cgp[2, :], c=Ugp[1, :], cmap=cmap, s=100)
    cb = fig0.colorbar(ax2.collections[0], ax=ax2)
    cb.ax.tick_params(labelsize=5)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Current Z-Disp')
    ax3.tick_params(labelsize=5)
    ax3.scatter(Cgp[0, :], Cgp[1, :], Cgp[2, :], c=Ugp[2, :], cmap=cmap, s=100)
    cb = fig0.colorbar(ax3.collections[0], ax=ax3)
    cb.ax.tick_params(labelsize=5)

    plt.show()

def plot_geo(np_n, np_e, u):
    plt.plot(u)
    plt.show()
    n_n    = int(len(np_n[:, 0]))

    # Node position matrices
    refX = np_n[:, 1:]
    curx = (refX.flatten() + u).reshape(n_n, 3)

    # Set figures
    fig0 = plt.figure()

    ax0 = fig0.add_subplot(121, projection='3d')
    ax1 = fig0.add_subplot(122, projection='3d')

    egs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 6), (6, 7), (7, 9), (9, 2),
             (7, 8), (8, 4)]
    
    # Plot patches
    for _, el_ns in enumerate(np_e):
        vtsx = []
        vtsX = []
        cNode = [el_ns[0]-1, el_ns[4]-1, el_ns[1]-1,
                        el_ns[5]-1, el_ns[2]-1, el_ns[6]-1,
                        el_ns[7]-1, el_ns[3]-1, el_ns[8]-1, 
                        el_ns[9]-1]
        vtsx = np.array([(xp, yp, zp) for xp, yp, zp in curx[cNode, :]])
        vtsX = np.array([(Xp, Yp, Zp) for Xp, Yp, Zp in refX[cNode, :]])
        for edge in egs:
            x = [vtsx[edge[0]][0], vtsx[edge[1]][0]]
            y = [vtsx[edge[0]][1], vtsx[edge[1]][1]]
            z = [vtsx[edge[0]][2], vtsx[edge[1]][2]]
            X = [vtsX[edge[0]][0], vtsX[edge[1]][0]]
            Y = [vtsX[edge[0]][1], vtsX[edge[1]][1]]
            Z = [vtsX[edge[0]][2], vtsX[edge[1]][2]]
            ax0.plot(X, Y, Z, c='k', alpha=1)
            ax1.plot(x, y, z, c='k', alpha=1)

    # Axis 
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Z')
    ax0.set_title('Reference Configuration')
    ax0.tick_params(labelsize=5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Current Configuration')
    ax1.tick_params(labelsize=5)

    plt.show()