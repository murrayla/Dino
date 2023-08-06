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
IJ = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
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
TMAP = {
        (0,0): 0, (1,1): 1, (2,2): 2, 
        (0,1): 3, (1,0): 3, (1,2): 4, 
        (2,1): 4, (2,0): 5, (0,2): 5
}

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

def cur_B_mat(e, np_n, np_e, x, dN):

    # Preallocate
    cur = np.zeros((N_EL_N, DIM))
    bmat = np.zeros((ORDER, DIM*2, N_EL_N*DIM))

    # Number of nodes and deformed x
    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)

    # Relevant node numebers for element
    rc = np_e[e, :]
    n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        cur[i, :] = xc[np.where(n_idx == local_node), :][0]

    # Loop through each column index c
    for q in range(0, ORDER, 1):
        # J = [∂x/∂ξ ∂y/∂ξ ∂z/∂ξ
        #      ∂x/∂η ∂y/∂η ∂z/∂η
        #      ∂x/∂ζ ∂y/∂ζ ∂z/∂ζ] @ Gauss
        jac = np.matmul(dN[q, :, :], cur).astype(np.float64)
        # dNdxyz = [∂φ1/∂x ∂φ2/∂x ... ∂φ10/∂x
        #           ∂φ1/∂y ∂φ2/∂y ... ∂φ10/∂y
        #           ∂φ1/∂z ∂φ2/∂z ... ∂φ10/∂z] @ Gauss   
        dNdxyz = np.matmul(np.linalg.inv(jac), dN[q, :, :])
        for r, c in enumerate(range(0, DIM*N_EL_N, DIM)):
            bmat[q, 0, c+0] = dNdxyz[0, r] # [∂φ/∂x,      0,      0] 
            bmat[q, 1, c+1] = dNdxyz[1, r] # [    0,  ∂φ/∂y,      0]
            bmat[q, 2, c+2] = dNdxyz[2, r] # [    0,      0,  ∂φ/∂z]
            bmat[q, 3, c+0] = dNdxyz[1, r] # [∂φ/∂y,  ...     ...
            bmat[q, 3, c+1] = dNdxyz[0, r] #   ...    ∂φ/∂x,      0]
            bmat[q, 4, c+1] = dNdxyz[2, r] # [    0,  ∂φ/∂z,  ...
            bmat[q, 4, c+2] = dNdxyz[1, r] #   ...    ...     ∂φ/∂y]
            bmat[q, 5, c+0] = dNdxyz[2, r] # [∂φ/∂z,  ...     ...
            bmat[q, 5, c+2] = dNdxyz[0, r] #   ...    ...     ∂φ/∂x]

    return bmat

def constitutive_eqs(e, c_vals, np_n, np_e, x, dN):

    # Number of nodes and deformed x
    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)
    
    # Preallocate 
    cur = np.zeros((N_EL_N, DIM))
    ref = np.zeros((N_EL_N, DIM))
    cau = np.zeros((ORDER, DIM, DIM))
    Dmat = np.zeros((ORDER, DIM*2, DIM*2))
    Fdef = np.zeros((ORDER, DIM, DIM))
    fdet = np.zeros(ORDER)

    # Relevant node numebrs for element
    rc = np_e[e, :]
    n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        cur[i, :] = xc[np.where(n_idx == local_node), :][0]
        ref[i, :] = np_n[np.where(n_idx == local_node), 1:][0]

    # Sub Gauss
    for q in range(0, ORDER, 1):
        # F = [∂x/∂X ∂x/∂Y ∂x/∂Z
        #      ∂y/∂X ∂y/∂Y ∂y/∂Z
        #      ∂z/∂X ∂z/∂Y ∂z/∂Z] @ Gauss
        Fdef[q, :, :] = np.matmul(
            np.matmul(
                np.linalg.inv(
                    np.matmul(dN[q, :, :], ref)
                ), dN[q, :, :]
            ), cur
        )
        # fdet = |F| @ Gauss
        fdet[q] = float(np.linalg.det(Fdef[q, :, :]))

    # Nearly incompressible
    d = 1000.10

    for n in range(0, ORDER, 1):
        # b = F*F^T @ Gauss
        b = np.matmul(Fdef[n, :, :], np.transpose(Fdef[n, :, :]))
        trb = np.trace(b)
        # Mooney Rivlin
        # W = c1(I1 - 3) + c2(I2-3) + d*(J-1)^2
        # Derivatives of Energy in terms of Invariants
        # First Order
        dWdI = np.array([c_vals[0], c_vals[1], 2*d*(fdet[n]-1)])
        # Second 
        ddWdII = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2*d]])
        # σ = [σ11, σ22, σ33, σ12, σ23, σ31]
        cau[n, :, :] = cauchy(dWdI, b, trb, fdet[n])
        # Jdijkl
        Dmat[n, :, :] = d_moduli(dWdI, ddWdII,  b, trb, fdet[n])
    
    return cau, Dmat

def cauchy(dWdI, b, trb, jac):
    # Preallocate
    cau = np.zeros((DIM, DIM))
    # Fill cauchy stress
    for i in range(0, DIM, 1):
        for j in range(0, DIM, 1):
            # [bij, (I * bij - bim * bmj), 0.5 * J * δij]
            dIdb = np.array(
                [
                    [
                        b[i, j], 
                        trb * b[i, j] - (b[i, 0]*b[0, j] + b[i, 1]*b[1, j] + b[i, 2]*b[2, j]), 
                        0.5 * jac * I[i, j]
                    ]
                ]
            )
            cau[i, j] = np.matmul(dIdb, dWdI)
    return 2/jac * cau

def d_moduli(dWdI, ddWdII,  b, trb, jac):

    # Preallocate
    d = np.zeros((DIM*2,DIM*2))
    # [4 * ∂W/∂II, ∂W/∂J]
    term4 = np.array(
        [
            [4*dWdI[0]], 
            [dWdI[0]]
        ]
    )
    for i in range(0, DIM, 1):
        for j in range(0, DIM, 1):
            # [bij, (I * bij - bim * bmj), 0.5 * J * δij]
            term1 = np.array(
                [
                    [
                        b[i, j], 
                        trb * b[i, j] - (b[i, 0]*b[0, j] + b[i, 1]*b[1, j] + b[i, 2]*b[2, j]), 
                        0.5 * jac * I[i, j]
                    ]
                ]
            )
            # 4 * term1 * [∂^2W/∂(I,II,J)^2]
            term1 = 4 * np.matmul(term1, ddWdII)
            for k in range(0, DIM, 1):
                for l in range(0, DIM, 1):
                    # [bkl, (I * bkl - bkm * bml), 0.5 * J * δkl]
                    term2 = np.array(
                        [
                            [b[k, l]],
                            [trb * b[k, l] - (b[k, 0]*b[0, l] + b[k, 1]*b[1, l] + b[k, 2]*b[2, l])],
                            [0.5 * jac * I[k, l]]
                        ]
                    )
                    # 0.5 * [δik * δjl + δil * δjk]
                    lil_b = 0.5 * (I[i, k] * I[j, l] + I[i, l] * I[j, k])
                    # [bij * bkl - 0.5 * (bik * bjl + bil * bjk), J * (δkl * δij - 2 * lil_b)]
                    term3 = np.array(
                        [
                            [
                                b[i, j] *  b[k, l] - 0.5 * (b[i, k] * b[j, l] + b[i, l] * b[j, k]),
                                jac * (I[i, j] * I[k, l] - 2*lil_b)
                            ]
                        ]
                    )
                    # dijkl = term1 * term2 + term3 * term4
                    d[TMAP[(i,j)], TMAP[(k,l)]] += np.matmul(term1, term2) + np.matmul(term3, term4)

    return d

def gauss_int(e, x, np_n, np_e, cau, bmat, dmat, kT, Fs, dN):

    # Number of nodes and deformed x
    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)
    
    # Preallocate 
    c = cau[e]
    b = bmat[e]
    d = dmat[e]
    cur = np.zeros((N_EL_N, DIM))

    # Relevant node numebers for element
    rc = np_e[e, :]
    n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        cur[i, :] = xc[np.where(n_idx == local_node), :][0]

    Kab = np.zeros((DIM*N_EL_N, DIM*N_EL_N))
    Gab = np.zeros((N_EL_N, N_EL_N))
    Fa = np.zeros((DIM, N_EL_N))

    for q, w in enumerate(WE):
        # J = [∂x/∂ξ ∂y/∂ξ ∂z/∂ξ
        #      ∂x/∂η ∂y/∂η ∂z/∂η
        #      ∂x/∂ζ ∂y/∂ζ ∂z/∂ζ] @ Gauss
        jac = np.matmul(dN[q, :, :], cur)
        # dNdxyz = [∂φ1/∂x ∂φ2/∂x ... ∂φ10/∂x
        #           ∂φ1/∂y ∂φ2/∂y ... ∂φ10/∂y
        #           ∂φ1/∂z ∂φ2/∂z ... ∂φ10/∂z] @ Gauss                
        dNdxyz = np.matmul(np.linalg.inv(jac), dN[q, :, :])
        # jdet = |J| @ Gauss
        jdet = abs(np.linalg.det(jac))
        # ============================== #
        # Solve Components
        # ============================== #
        # Material Stiffness Km = ∫ BαT * DT * Bβ * J dv
        Kab += np.matmul(
            np.matmul(
                np.transpose(b[q, :, :]), d[q, :, :]
            ), b[q, :, :]
        ) * jdet * w
        # Geometric / Initial Stiffness Gαβ = ∫ Nα,i * σij * Nβ,j * J dv 
        Gab += np.matmul(
            np.transpose(dNdxyz), np.matmul(c[q, :, :], dNdxyz)
        ) * jdet * w 
        # Residual Fα = - ∫ BαTσ * J dv OR Fα = - ∫ σ * ∂Nα/∂x * J
        Fa += np.matmul(c[q, :, :], dNdxyz) * jdet * w

    Fa *= (-1) 

    # Fill Fsol & Ftan
    # Loop rows
    for i in range(0, N_EL_N, 1):
        # Loop columns
        for j in range(0, N_EL_N, 1):
            # Allocate to Tangent
            kT[
                DIM*(rc[i]-1):DIM*(rc[i]-1)+3, DIM*(rc[j]-1):DIM*(rc[j]-1)+3
            ] += (Kab[DIM*i:DIM*i+DIM, DIM*j:DIM*j+DIM] + Gab[i, j] * I)
        # Allocate to Residual
        Fs[DIM*(rc[i]-1):DIM*(rc[i]-1)+DIM] += Fa[:, i]

    return Fs, kT

def nonlinear_solve(x, np_n, np_e, dN, c_vals, n_ele, num_pro):

    # Preallocate
    n_n = int(len(np_n[:, 0]))
    e_n = range(0, n_ele, 1)
    Fs = np.zeros(DIM*n_n)
    kT = np.zeros((DIM*n_n, DIM*n_n))
   
    Bmat_Pool = mp.Pool(processes=num_pro)
    p_Bmat = partial(cur_B_mat, np_n=np_n, np_e=np_e, x=x, dN=dN)
    Bmats = Bmat_Pool.map(p_Bmat, e_n)

    sig_Pool = mp.Pool(processes=num_pro)
    p_sig = partial(constitutive_eqs, c_vals=c_vals, np_n=np_n, np_e=np_e, x=x, dN=dN)
    sig_Results = sig_Pool.map(p_sig, e_n)
    cau, Dmats = zip(*sig_Results)

    gau_Pool = mp.Pool(processes=num_pro)
    g_sig = partial(gauss_int, x=x, np_n=np_n, np_e=np_e, cau=cau, bmat=Bmats, \
                    dmat=Dmats, kT=kT, Fs=Fs, dN=dN)
    gau_Results = gau_Pool.map(g_sig, e_n)
    Fsol_Results, Ftan_Results = zip(*gau_Results)

    return sum(Fsol_Results), sum(Ftan_Results)

def newton_raph(u, nodes, np_n, np_e, n_ele, dN, c_vals, num_pro, iters, tol):
    xn = np_n[:, 1:].flatten() + u 
    for i in range(0, iters, 1):
        nrFunc, nrFtan = nonlinear_solve(xn, np_n, np_e, dN, c_vals, n_ele, num_pro)
        nrFtanSol = np.copy(nrFtan)
        if nodes != None:
            nrFunc[nodes] = 0
            for idx in nodes:
                nrFtanSol[idx, :] = 0
                nrFtanSol[:, idx] = 0
                nrFtanSol[idx, idx] = nrFtan[idx, idx]
        un = np.matmul(np.linalg.inv(nrFtanSol), nrFunc)
        xn1 = xn + un
        xn = xn1

        SSE = sum(np.square(nrFunc))

        print("Residual Average: {}".format(SSE))
        print("Iteration Number: {}".format(i))

        if SSE < tol:
            return xn1, i

    print("Did not converge")
    return xn, iters

def plot_disps(np_n, np_e, u, n_ele, phi):
    plt.plot(u)
    plt.show()
    cmap = get_cmap('seismic')
    # cmap = get_cmap('Blues')

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

        for q in range(0, ORDER, 1):
            x_gp[e*len(GP)-1 + q] = np.matmul(np.transpose(phi[q, :]), xyz[:, 0]) + np.matmul(np.transpose(phi[q, :]), uvw[:, 0])
            y_gp[e*len(GP)-1 + q] = np.matmul(np.transpose(phi[q, :]), xyz[:, 1]) + np.matmul(np.transpose(phi[q, :]), uvw[:, 1])
            z_gp[e*len(GP)-1 + q] = np.matmul(np.transpose(phi[q, :]), xyz[:, 2]) + np.matmul(np.transpose(phi[q, :]), uvw[:, 2])
            u_gp[e*len(GP)-1 + q] = np.matmul(np.transpose(phi[q, :]), uvw[:, 0])
            v_gp[e*len(GP)-1 + q] = np.matmul(np.transpose(phi[q, :]), uvw[:, 1])
            w_gp[e*len(GP)-1 + q] = np.matmul(np.transpose(phi[q, :]), uvw[:, 2])
    
    Ugp = np.array([u_gp, v_gp, w_gp]) 
    Cgp = np.array([x_gp, y_gp, z_gp]) 

    n_n = int(len(np_n[:, 0]))

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