import numpy as np
import sympy as sym
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from functools import partial

DIM = 3
N_EL_N = 10
ORDER = 5

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

def element_assign():
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
    delPhi = sym.Matrix(
        [
            [sym.diff(phi[j], xi, 1) for j in range(0, N_EL_N, 1)],  
            [sym.diff(phi[k], eta, 1) for k in range(0, N_EL_N, 1)],
            [sym.diff(phi[m], zeta, 1) for m in range(0, N_EL_N, 1)]
        ]
    )
    
    return (xi, eta, zeta), phi, delPhi

def gauss_num_int():

    we = np.array([-4/5, 9/20, 9/20, 9/20, 9/20])
    gp = np.array(
        [
            [1/4, 1/4, 1/4], 
            [1/2, 1/6, 1/6], 
            [1/6, 1/2, 1/6], 
            [1/6, 1/6, 1/2],
            [1/6, 1/6, 1/6]
        ]
    )
    return we, gp

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

def deformation_gradient(X, x):

    xez, _, delPhi = element_assign()

    _, gp = gauss_num_int()

    Fdef = np.zeros((N_EL_N, DIM, DIM))
    detF = np.zeros(N_EL_N)

    x = sym.Matrix(x)
    X = sym.Matrix(X)

    # Determine δ{xyz}/δ{xietazeta} Jacobians 
    jxyz = delPhi * x
    # Determine δ{XYZ}/δ{xietazeta} Jacboain and δ{uv}/δ{xy}
    jXYZ = delPhi * X
    jXEZ = jXYZ.inv()
    dxyzdXYZ = jXEZ * jxyz

    n_ps = np.array(
        [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5],
            [0.5, 0, 0.5], [0.5, 0, 0], [0, 0, 0.5],
            [0, 0.5, 0]
        ]
    )

    for i, p in enumerate(n_ps):
        Fdef[i, :, :] = np.array(dxyzdXYZ.subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]})).astype(float)
        detF[i] = np.linalg.det(Fdef[i, :, :])

    detF = np.array(detF).astype(float)

    return Fdef, detF

def ref_B_mat(e, np_n, np_e, x):

    xez, _, delPhi = element_assign()
    _, gp = gauss_num_int()

    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)

    # Preallocate natural coordinates
    X_ele = np.zeros((N_EL_N, DIM))
    x_ele = np.zeros((N_EL_N, DIM))

    # Set rows and columns GLOBAL
    rc = np_e[e, :]

    # Find coordinates for those nodes
    np_n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        n_idx = np.where(np_n_idx == local_node)[0]
        X_ele[i] = np_n[n_idx, 1:DIM+1][0]
        x_ele[i] = xc[n_idx, :][0]

    #########

    Fdef = np.zeros((N_EL_N, DIM, DIM))

    x = sym.Matrix(x_ele)
    Xr = sym.Matrix(X_ele)

    # Determine δ{xyz}/δ{xietazeta} Jacobians 
    jxyz = delPhi * x
    # Determine δ{XYZ}/δ{xietazeta} Jacboain and δ{uv}/δ{xy}
    jXYZ = delPhi * Xr
    jXEZ = jXYZ.inv()
    dxyzdXYZ = jXEZ * jxyz

    n_ps = np.array(
        [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5],
            [0.5, 0, 0.5], [0.5, 0, 0], [0, 0, 0.5],
            [0, 0.5, 0]
        ]
    )

    for i, p in enumerate(n_ps):
        Fdef[i, :, :] = np.array(dxyzdXYZ.subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]})).astype(float)

    ##########

    # Jacobian
    jac = delPhi * X_ele
    detJ = jac.det()
    b_sub = jac.inv() * delPhi

    # Initialize a matrix filled with zeros
    b_mat = sym.zeros(ORDER*DIM*2, N_EL_N*DIM)
    jdet = sym.zeros(ORDER, 1)

    # Loop through each column index c
    for i, p in enumerate(gp):
        for r, c in enumerate(range(0, DIM*N_EL_N, DIM)):
            f = Fdef[r, :, :]
            # [F11φα,1 F21φα,1 F31φα,1] 
            b_mat[ORDER*i+0, c+0] = f[0,0] * b_sub[0, r]
            b_mat[ORDER*i+0, c+1] = f[1,0] * b_sub[0, r]
            b_mat[ORDER*i+0, c+2] = f[2,0] * b_sub[0, r]
            # [F12φα,2 F22φα,2 F32φα,2]
            b_mat[ORDER*i+1, c+0] = f[0,1] * b_sub[1, r]
            b_mat[ORDER*i+1, c+1] = f[1,1] * b_sub[1, r]
            b_mat[ORDER*i+1, c+2] = f[2,1] * b_sub[1, r]
            # [F13φα,3 F23φα,3 F33φα,3]
            b_mat[ORDER*i+2, c+0] = f[0,2] * b_sub[2, r]
            b_mat[ORDER*i+2, c+1] = f[1,2] * b_sub[2, r]
            b_mat[ORDER*i+2, c+2] = f[2,2] * b_sub[2, r]
            # [F11φα,2 + F12φα,1 F21φα,2 + F22φα,1 F31φα,2 + F32φα,1]
            b_mat[ORDER*i+3, c+0] = f[0,0] * b_sub[1, r] + f[0,1] * b_sub[0, r]
            b_mat[ORDER*i+3, c+1] = f[1,0] * b_sub[1, r] + f[1,1] * b_sub[0, r]
            b_mat[ORDER*i+3, c+2] = f[2,0] * b_sub[1, r] + f[2,1] * b_sub[0, r]
            # [F12φα,3 + F13φα,2 F22φα,3 + F23φα,2 F32φα,3 + F33φα,2]
            b_mat[ORDER*i+4, c+0] = f[0,1] * b_sub[2, r] + f[0,2] * b_sub[1, r]
            b_mat[ORDER*i+4, c+1] = f[1,1] * b_sub[2, r] + f[1,2] * b_sub[1, r]
            b_mat[ORDER*i+4, c+2] = f[2,1] * b_sub[2, r] + f[2,2] * b_sub[1, r]
            # [F13φα,1 + F11φα,3 F23φα,1 + F21φα,3 F33φα,1 + F31φα,3]  
            b_mat[ORDER*i+5, c+0] = f[0,2] * b_sub[0, r] + f[0,0] * b_sub[2, r]
            b_mat[ORDER*i+5, c+1] = f[1,2] * b_sub[0, r] + f[1,0] * b_sub[2, r]
            b_mat[ORDER*i+5, c+2] = f[2,2] * b_sub[0, r] + f[2,0] * b_sub[2, r]

        b_mat[ORDER*i:ORDER*i+6, :] = b_mat[ORDER*i:ORDER*i+6, :].subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]})
        jdet[i] = abs(detJ.subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]}))

    return np.array(b_mat).astype(float), np.array(jdet).astype(float)

def constitutive_eqs(e, c_vals, np_n, np_e, x):

    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)

    # Preallocate natural coordinates
    X_ele = np.zeros((N_EL_N, DIM))
    x_ele = np.zeros((N_EL_N, DIM))
    Smat = np.zeros((DIM*2, ORDER))
    Dmat = np.zeros((ORDER, DIM*2, DIM*2))

    # Set rows and columns GLOBAL
    rc = np_e[e, :]

    # Find coordinates for those nodes
    np_n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        n_idx = np.where(np_n_idx == local_node)[0]
        X_ele[i] = np_n[n_idx, 1:DIM+1][0]
        x_ele[i] = xc[n_idx, :][0]

    ######
    xez, _, delPhi = element_assign()

    _, gp = gauss_num_int()
    Fdef = np.zeros((ORDER, DIM, DIM))
    detF = np.zeros((ORDER, 1))

    x = sym.Matrix(x_ele)
    Xr = sym.Matrix(X_ele)

    # Determine δ{xyz}/δ{xietazeta} Jacobians 
    jxyz = delPhi * x
    # Determine δ{XYZ}/δ{xietazeta} Jacboain and δ{uv}/δ{xy}
    jXYZ = delPhi * Xr
    jXEZ = jXYZ.inv()
    dxyzdXYZ = jXEZ * jxyz

    for i, p in enumerate(gp):
        Fdef[i, :, :] = np.array(dxyzdXYZ.subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]})).astype(float)
        detF[i] = np.linalg.det(Fdef[i, :, :])

    detF = np.array(detF).astype(float)
    
    ######
    d = 1000.10

    for n in range(0, ORDER, 1):
        f = Fdef[n, :, :]
        jac = detF[n, 0]
        c = f.T * f
        invC = np.linalg.inv(c)
        trC = c[0,0] + c[1,1] + c[2,2]
        # Mooney Rivlin
        # W = c1(I1 - 3) + c2(I2-3) + d*(J-1)^2
        # Derivatives of Energy in terms of Invariants
        # First Order
        dWdI = np.array([c_vals[0], c_vals[1], 2*d*(jac-1)])
        # Second Order
        ddWdII = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2*d]])
        Smat[:, n] = second_piola(dWdI, c, invC, trC, jac)
        Dmat[n, :, :] = elastic_moduli(dWdI, ddWdII, c, invC, trC, jac)
    
    print(Smat)
    return Smat, Dmat

def second_piola(dWdI, C, invC, trC, Jc):
    dd = np.eye(DIM)
    s = np.zeros((DIM, DIM))
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
    for _, (i, j) in enumerate(ij):
        delIdelC = np.array(
            [
                [dd[i, j], trC * dd[i, j] - C[i, j], 0.5 * Jc * invC[i, j]]
            ]
        )
        sPk = np.matmul(delIdelC, dWdI)
        s[i, j] = 2 * sPk
    Spk = np.array([s[0,0], s[1,1], s[2,2], s[0,1], s[1,2], s[2,0]])
    return Spk

def elastic_moduli(dWdI, ddWdII, C, invC, trC, Jc):
    dd = np.eye(DIM)
    moduli = np.zeros((DIM*2,DIM*2))
    # Stress Indexes
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
    for r, (i, j) in enumerate(ij):
        term1 = np.array(
            [
                [
                    dd[i, j], 
                    trC * dd[i, j] - C[i, j], 
                    0.5 * Jc * invC[i, j]
                ]
            ]
        )
        term1 = np.matmul(term1, ddWdII)
        term4 = np.array(
            [
                [4*dWdI[0]], 
                [dWdI[0]]
            ]
        )
        for c, (k, l) in enumerate(ij):
            term2 = np.array(
                [
                    [dd[k, l]],
                    [trC * dd[k, l] - C[k, l]],
                    [0.5 * Jc * invC[k, l]]
                ]
            )
            lil_c = 0.5 * (invC[i, k] * invC[j, l] +
                           invC[i, l] * invC[j, k])
            term3 = np.array(
                [
                    [
                        dd[i, k] * dd[k, l] - 0.5 * (dd[i, k] * dd[j, l] + dd[i, l] * dd[j, k]),
                        Jc * (invC[i, j] * invC[k, l] - 2*lil_c)
                    ]
                ]
            )
            moduli[r, c] = 4 * np.matmul(term1, term2) + np.matmul(term3, term4)
    return moduli

def geometric_tangent_k(e, Spk, np_e, k_geo, detJ):

    xez, _, delPhi = element_assign()
    we, gp = gauss_num_int()

    S = Spk[e]
    jac = detJ[e]
    rc = np_e[e, :]
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
    Idn = np.eye(DIM)

    for a in range(0, N_EL_N, 1):
        for b in range(0, N_EL_N, 1):
            Gab = sym.zeros(ORDER, 1)
            g = 0
            for n, (i, j) in enumerate(ij):
                Gab[0] += delPhi[i, a] * S[n, 0] * delPhi[j, b]
                Gab[1] += delPhi[i, a] * S[n, 1] * delPhi[j, b]
                Gab[2] += delPhi[i, a] * S[n, 2] * delPhi[j, b]
                Gab[3] += delPhi[i, a] * S[n, 3] * delPhi[j, b]
                Gab[4] += delPhi[i, a] * S[n, 4] * delPhi[j, b]
            for q, w in enumerate(we):
                g += w * jac[q] * float((Gab[q]).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            k_geo[DIM*(rc[a]-1):DIM*(rc[a]-1)+3, DIM*(rc[b]-1):DIM*(rc[b]-1)+3] += g * Idn

    return k_geo
    
def gauss_int_fsol(e, Bmat, detJ, Smat, k_sol, np_e):

    we, _ = gauss_num_int()

    bT = np.transpose(Bmat[e])
    s = Smat[e]
    jac = detJ[e]
    rc = np_e[e, :]
    term = np.zeros(DIM*N_EL_N)

    for i in range(0, N_EL_N, 1):
        for q, w in enumerate(we):
            bTs = -1 * jac[q, 0] * np.matmul(bT[DIM*i:DIM*i+3, q*ORDER:q*ORDER+6], s[:, q])
            term[DIM*i:DIM*i+3] += w * bTs

        k_sol[DIM*(rc[i]-1):DIM*(rc[i]-1)+3] += term[DIM*i:DIM*i+3]

    return k_sol

def gauss_int_ftan(e, Bmat, detJ, Dmat, k_tan, np_e):

    we, _ = gauss_num_int()

    b = Bmat[e]
    bT = np.transpose(b)
    d = Dmat[e]
    jac = detJ[e]

    rc = np_e[e, :]
    term_tan = np.zeros((DIM*N_EL_N, DIM*N_EL_N))

    # Gauss Quadrature
    for i in range(0, N_EL_N, 1):
        for j in range(0, N_EL_N, 1):
            for q, w in enumerate(we):
                term = np.matmul(np.matmul(bT[DIM*i:DIM*i+3, q*ORDER:q*ORDER+6], d[q, :, :]), b[q*ORDER:q*ORDER+6, DIM*i:DIM*i+3]) * jac[q]
                term_tan[DIM*i:DIM*i+3, DIM*j:DIM*j+3] += w * term

            k_tan[DIM*(rc[i]-1):DIM*(rc[i]-1)+3, DIM*(rc[j]-1):DIM*(rc[j]-1)+3] += term_tan[DIM*i:DIM*i+3, DIM*j:DIM*j+3]

    return k_tan

def nonlinear_solve(u, np_n, np_e, c_vals, n_ele, num_pro):

    n_n = int(len(np_n[:, 0]))
    x = np_n[:, 1:].flatten() 
    x = x + u

    k_sol = np.zeros(DIM*n_n)
    k_tan = np.zeros((DIM*n_n, DIM*n_n))
    k_geo = np.zeros((DIM*n_n, DIM*n_n))

    Bmat_Pool = mp.Pool(processes=num_pro)
    Smat_Pool = mp.Pool(processes=num_pro)
    Gmat_Pool = mp.Pool(processes=num_pro)
    Fsol_Pool = mp.Pool(processes=num_pro)
    Ftan_Pool = mp.Pool(processes=num_pro)

    ele_list = range(0, n_ele, 1)

    p_Bmat = partial(ref_B_mat, np_n=np_n, np_e=np_e, x=x)
    p_Smat = partial(constitutive_eqs, c_vals=c_vals, np_n=np_n, np_e=np_e, x=x)

    Bmat_Results = Bmat_Pool.map(p_Bmat, ele_list)
    Bmats, detJs = zip(*Bmat_Results)
    Smat_Results = Smat_Pool.map(p_Smat, ele_list)
    Smats, Dmats = zip(*Smat_Results)

    p_Gmat = partial(geometric_tangent_k, Spk=Smats, np_e=np_e, k_geo=k_geo, detJ=detJs)
    Kgeos = Gmat_Pool.map(p_Gmat, ele_list)
    Gmat = sum(Kgeos)

    p_Fsol = partial(gauss_int_fsol, Bmat=Bmats, detJ=detJs, Smat=Smats, k_sol=k_sol, np_e=np_e)
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
    _, gp = gauss_num_int()
    x_gp = np.zeros(n_ele*len(gp))
    y_gp = np.zeros(n_ele*len(gp))
    z_gp = np.zeros(n_ele*len(gp))
    u_gp = np.zeros(n_ele*len(gp))
    v_gp = np.zeros(n_ele*len(gp))
    w_gp = np.zeros(n_ele*len(gp))

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

        for q, p in enumerate(gp):
            x_gp[e*len(gp)-1 + q] = float((u_sf[0]+x[0]).subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]}))
            y_gp[e*len(gp)-1 + q] = float((v_sf[0]+y[0]).subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]}))
            z_gp[e*len(gp)-1 + q] = float((w_sf[0]+z[0]).subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]}))
            u_gp[e*len(gp)-1 + q] = float((u_sf[0]).subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]}))
            v_gp[e*len(gp)-1 + q] = float((v_sf[0]).subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]}))
            w_gp[e*len(gp)-1 + q] = float((w_sf[0]).subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]}))
    
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