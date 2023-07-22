import numpy as np
import sympy as sym
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from functools import partial

def apply_nonlinear_BC(np_n, u, nodes, BC_0, BC_1, axi, dim=3):

    min_val = np.amin(np_n[:, axi+1])
    max_val = np.amax(np_n[:, axi+1])

    # Apply Bounadry Conditions
    # Check position of node and apply BC if condition met
    for n in np_n[:, 0]:
        n_idx = int(n)
        n_val = np_n[np_n[:, 0] == n, axi+1][0]

        if n_val == min_val and BC_0 is not None:
            u[dim*(n_idx-1)+axi] = BC_0
            nodes.append(dim*(n_idx-1)+axi)

        elif n_val == max_val and BC_1 is not None:
            u[dim*(n_idx-1)+axi] = BC_1
            nodes.append(dim*(n_idx-1)+axi)

    return u, nodes

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
                        [sym.diff(phi[j], eta, 1)  for j in range(0, n_el_n, 1)],
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

def deformation_gradient(X, x, el_type, el_order, order):

    dim, n_el_n, xez, _, delPhi = element_assign(el_type, el_order)

    Fdef = np.zeros((n_el_n, 3, 3))
    detF = np.zeros(n_el_n)

    n_ps = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5]
            ]

    x = sym.Matrix(x)
    X = sym.Matrix(X)
    
    # Determine δ{xy}/δ{xieta} Jacobians and inverse
    jac_xyz = delPhi * x
    # Determine δ{uv}/δ{xieta} Jacboain and δ{uv}/δ{xy}
    jac_XYZ = delPhi * X
    inv_jac_XYZ = jac_XYZ.inv()
    dxyzdXYZ = inv_jac_XYZ * jac_xyz

    for i, p in enumerate(n_ps):
        Fdef[i, :, :] = np.array(dxyzdXYZ.subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]})).astype(float)
        detF[i] = np.linalg.det(Fdef[i, :, :])

    detF = np.array(detF).astype(float)

    return Fdef, detF

def ref_B_mat(e, np_n, np_e, x, el_type, el_order, order):

    dim, n_el_n, _, _, delPhi = element_assign(el_type, el_order)

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

    F, _ = deformation_gradient(X_ele, x_ele, el_type, el_order, order)

    # Jacobian
    jac = delPhi * X_ele
    detJ = jac.det()

    # Initialize a matrix filled with zeros
    b_mat = sym.zeros(6, n_el_n*dim)

    # Loop through each column index c
    for r, c in enumerate(range(0, dim*n_el_n, dim)):
        Fdef = F[r, :, :]
        # [F11φα,1 F21φα,1 F31φα,1] 
        b_mat[0, c+0] = Fdef[0,0] * delPhi[0, r]
        b_mat[0, c+1] = Fdef[1,0] * delPhi[0, r]
        b_mat[0, c+2] = Fdef[2,0] * delPhi[0, r]
        # [F12φα,2 F22φα,2 F32φα,2]
        b_mat[1, c+0] = Fdef[0,1] * delPhi[1, r]
        b_mat[1, c+1] = Fdef[1,1] * delPhi[1, r]
        b_mat[1, c+2] = Fdef[2,1] * delPhi[1, r]
        # [F13φα,3 F23φα,3 F33φα,3]
        b_mat[2, c+0] = Fdef[0,2] * delPhi[2, r]
        b_mat[2, c+1] = Fdef[1,2] * delPhi[2, r]
        b_mat[2, c+2] = Fdef[2,2] * delPhi[2, r]
        # [F11φα,2 + F12φα,1 F21φα,2 + F22φα,1 F31φα,2 + F32φα,1]
        b_mat[3, c+0] = Fdef[0,0] * delPhi[1, r] + Fdef[0,1] * delPhi[0, r]
        b_mat[3, c+1] = Fdef[1,0] * delPhi[1, r] + Fdef[1,1] * delPhi[0, r]
        b_mat[3, c+2] = Fdef[2,0] * delPhi[1, r] + Fdef[2,1] * delPhi[0, r]
        # [F12φα,3 + F13φα,2 F22φα,3 + F23φα,2 F32φα,3 + F33φα,2]
        b_mat[4, c+0] = Fdef[0,1] * delPhi[2, r] + Fdef[0,2] * delPhi[1, r]
        b_mat[4, c+1] = Fdef[1,1] * delPhi[2, r] + Fdef[1,2] * delPhi[1, r]
        b_mat[4, c+2] = Fdef[2,1] * delPhi[2, r] + Fdef[2,2] * delPhi[1, r]
        # [F13φα,1 + F11φα,3 F23φα,1 + F21φα,3 F33φα,1 + F31φα,3]  
        b_mat[5, c+0] = Fdef[0,2] * delPhi[0, r] + Fdef[0,0] * delPhi[2, r]
        b_mat[5, c+1] = Fdef[1,2] * delPhi[0, r] + Fdef[1,0] * delPhi[2, r]
        b_mat[5, c+2] = Fdef[2,2] * delPhi[0, r] + Fdef[2,0] * delPhi[2, r]

    return b_mat, detJ

def constitutive_eqs(e, con_type, c_vals, np_n, np_e, x, el_type, el_order, order):

    dim, n_el_n, _, _, _ = element_assign(el_type, el_order)

    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, 3)

    # Preallocate natural coordinates
    X_ele = np.zeros((n_el_n, dim))
    x_ele = np.zeros((n_el_n, dim))
    Smat = np.zeros((6, n_el_n))
    Dmat = np.zeros((n_el_n, 6, 6))

    # Set rows and columns GLOBAL
    rc = np_e[e, :]

    # Find coordinates for those nodes
    np_n_idx = np_n[:, 0]
    for i, local_node in enumerate(rc):
        n_idx = np.where(np_n_idx == local_node)[0]
        X_ele[i] = np_n[n_idx, 1:dim+1][0]
        x_ele[i] = xc[n_idx, :][0]

    F, J = deformation_gradient(X_ele, x_ele, el_type, el_order, order)

    for n in range(0, n_el_n, 1):
        Fdef = F[n, :, :]
        detF = J[n]
        Cgre = Fdef.T * Fdef
        # Mooney Rivlin
        # W = c1(I1 - 3) + c2(I2-3)
        if con_type == 0:
            # Derivatives of Energy in terms of Invariants
            # First Order
            dWdI = [c_vals[0], c_vals[1], 0]
            # Second Order
            ddWdII = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            Smat[:, n] = second_piola(dWdI, Cgre, detF)
            Dmat[n, :, :] = elastic_moduli(dWdI, ddWdII, Cgre, detF)

    return Smat, Dmat

def second_piola(dWdI, C, Jc):
    sPK = np.zeros(6)
    dd = np.eye(3)
    # Stress Indexes
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
    invC = np.linalg.inv(C)
    # Fill 2ndPK
    for i in range(0, 6, 1): 
        delIdelC = [dd[ij[i,0], ij[i,1]],
                    np.trace(C) * dd[ij[i,0], ij[i,1]] - C[ij[i,0], ij[i,1]],
                    0.5 * Jc * invC[ij[i,0], ij[i,1]]]
        sPk_i = np.matmul(delIdelC, np.array(dWdI))
        sPK[i] = 2 * sPk_i
    return np.transpose(sPK)

def elastic_moduli(dWdI, ddWdII, C, Jc):
    dd = np.eye(3)
    invC = np.linalg.inv(C)
    moduli = np.zeros((6,6))
    # Stress Indexes
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
    for r in range(0, 6, 1):
        term1 = np.matmul(
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
            moduli[r, c] = 4*np.matmul(term1, term3) + np.matmul(term4, term2)
    return moduli

def geometric_tangent_k(e, Spk, np_e, k_geo, detJ, el_type, el_order, order):

    dim, n_el_n, xez, _, delPhi = element_assign(el_type, el_order)
    we, gp = gauss_num_int(el_type, order)

    Spk = Spk[e]
    J = detJ[e]
    rc = np_e[e, :]
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
    Idn = np.eye(3)

    for a in range(0, n_el_n, 1):
        S = Spk[:, a]
        G = 0
        for b in range(0, n_el_n, 1):
            Gab = delPhi[ij[0,0], a] * S[0] * delPhi[ij[0,1], b] + \
                    delPhi[ij[1,0], a] * S[1] * delPhi[ij[1,1], b] + \
                    delPhi[ij[2,0], a] * S[2] * delPhi[ij[2,1], b] + \
                    delPhi[ij[3,0], a] * S[3] * delPhi[ij[3,1], b] + \
                    delPhi[ij[4,0], a] * S[4] * delPhi[ij[4,1], b] + \
                    delPhi[ij[5,0], a] * S[5] * delPhi[ij[5,1], b]
            for q, w in enumerate(we):
                G += w + float((Gab*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            k_geo[dim*(rc[a]-1):dim*(rc[a]-1)+3, dim*(rc[b]-1):dim*(rc[b]-1)+3] += G * Idn

    return k_geo
    
def gauss_int_fsol(e, Bmat, detJ, Smat, k_sol, np_e, el_type, el_order, order):
     
    dim, n_el_n, xez, _, _ = element_assign(el_type, el_order)
    we, gp = gauss_num_int(el_type, order)

    Bmat = Bmat[e]
    BTra = np.transpose(Bmat)
    Smat = Smat[e]
    J = detJ[e]
    rc = np_e[e, :]

    term = np.zeros(dim*n_el_n)

    for i in range(0, n_el_n, 1):
        BtS = -1*np.matmul(BTra[3*i:3*i+3, :], Smat[:, i]) 
        for q, w in enumerate(we):
            term[dim*i + 0] += w * float((BtS[0]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term[dim*i + 1] += w * float((BtS[1]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term[dim*i + 2] += w * float((BtS[2]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))

        k_sol[dim*(rc[i]-1)+0] += term[dim*i + 0]
        k_sol[dim*(rc[i]-1)+1] += term[dim*i + 1]
        k_sol[dim*(rc[i]-1)+2] += term[dim*i + 2]

    return k_sol

def gauss_int_ftan(e, Bmat, detJ, Dmat, Gmat, k_tan, np_e, el_type, el_order, order):

    dim, n_el_n, xez, _, _ = element_assign(el_type, el_order)
    we, gp = gauss_num_int(el_type, order)

    Bmat = Bmat[e]
    BTra = Bmat.T
    Dmat = Dmat[e]
    J = detJ[e]

    rc = np_e[e, :]
    term_tan = np.zeros((Bmat.shape[1], Bmat.shape[1]))

    # Gauss Quadrature
    for i in range(0, n_el_n, 1):
        D = Dmat[i, :, :]
        term = np.matmul(BTra[3*i:3*i+3, :], D) 
        term = np.matmul(term, Bmat[:, 3*i:3*i+3])
        for j in range(0, n_el_n, 1):
            for q, w in enumerate(we):
                # [1 2 3]
                term_tan[dim*i+0, dim*j+0] += w * float((term[0, 0]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
                term_tan[dim*i+0, dim*j+1] += w * float((term[0, 1]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
                term_tan[dim*i+0, dim*j+2] += w * float((term[0, 2]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
                # [4 5 6]
                term_tan[dim*i+1, dim*j+0] += w * float((term[1, 0]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
                term_tan[dim*i+1, dim*j+1] += w * float((term[1, 1]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
                term_tan[dim*i+1, dim*j+2] += w * float((term[1, 2]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
                # [4 5 6]
                term_tan[dim*i+2, dim*j+0] += w * float((term[2, 0]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
                term_tan[dim*i+2, dim*j+1] += w * float((term[2, 1]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
                term_tan[dim*i+2, dim*j+2] += w * float((term[2, 2]*J).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))

            term_tan[dim*i:dim*i+3, dim*i:dim*i+3] += Gmat[dim*(rc[i]-1):dim*(rc[i]-1)+3, dim*(rc[j]-1):dim*(rc[j]-1)+3]
            # [1 2 3]
            k_tan[dim*(rc[i]-1)+0, dim*(rc[j]-1)+0] += term_tan[dim*i+0, dim*j+0] 
            k_tan[dim*(rc[i]-1)+0, dim*(rc[j]-1)+1] += term_tan[dim*i+0, dim*j+1] 
            k_tan[dim*(rc[i]-1)+0, dim*(rc[j]-1)+2] += term_tan[dim*i+0, dim*j+2] 
            # [4 5 6]
            k_tan[dim*(rc[i]-1)+1, dim*(rc[j]-1)+0] += term_tan[dim*i+1, dim*j+0] 
            k_tan[dim*(rc[i]-1)+1, dim*(rc[j]-1)+1] += term_tan[dim*i+1, dim*j+1] 
            k_tan[dim*(rc[i]-1)+1, dim*(rc[j]-1)+2] += term_tan[dim*i+1, dim*j+2] 
            # [7 8 9]
            k_tan[dim*(rc[i]-1)+2, dim*(rc[j]-1)+0] += term_tan[dim*i+2, dim*j+0] 
            k_tan[dim*(rc[i]-1)+2, dim*(rc[j]-1)+1] += term_tan[dim*i+2, dim*j+1] 
            k_tan[dim*(rc[i]-1)+2, dim*(rc[j]-1)+2] += term_tan[dim*i+2, dim*j+2] 

    return k_tan

def nonlinear_solve(u, np_n, np_e, con_type, c_vals, n_ele, num_pro, el_type, el_order, order):

    dim, _, _, _, _ = element_assign(el_type, el_order)

    n_n = int(len(np_n[:, 0]))
    x = np_n[:, 1:].flatten() 
    x = x + u

    k_sol = np.zeros(dim*n_n)
    k_tan = np.zeros((dim*n_n, dim*n_n))
    k_geo = np.zeros((dim*n_n, dim*n_n))

    Bmat_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Smat_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Gmat_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Fsol_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)
    Ftan_Pool = mp.Pool(processes=num_pro, maxtasksperchild=100)

    ele_list = range(0, n_ele, 1)

    p_Bmat = partial(ref_B_mat, np_n=np_n, np_e=np_e, x=x, el_type=el_type, \
                     el_order=el_order, order=order)
    p_Smat = partial(constitutive_eqs, con_type=con_type, c_vals=c_vals, \
                     np_n=np_n, np_e=np_e, x=x, el_type=el_type, el_order=el_order, order=order)

    Bmat_Results = Bmat_Pool.map(p_Bmat, ele_list)
    Bmats, detJs = zip(*Bmat_Results)
    Smat_Results = Smat_Pool.map(p_Smat, ele_list)
    Smats, Dmats = zip(*Smat_Results)

    p_Gmat = partial(geometric_tangent_k, Spk=Smats, np_e=np_e, k_geo=k_geo, \
                     detJ=detJs, el_type=el_type, el_order=el_order, order=order)
    Kgeos = Gmat_Pool.map(p_Gmat, ele_list)
    Gmat = sum(Kgeos)

    p_Fsol = partial(gauss_int_fsol, Bmat=Bmats, detJ=detJs, Smat=Smats, \
                        k_sol=k_sol, np_e=np_e, el_type=el_type, el_order=el_order, order=order)
    
    p_Ftan = partial(gauss_int_ftan, Bmat=Bmats, detJ=detJs, Dmat=Dmats, Gmat=Gmat, \
                        k_tan=k_tan, np_e=np_e, el_type=el_type, el_order=el_order, order=order)
    
    Fsol_Results = Fsol_Pool.map(p_Fsol, ele_list)
    nrFunc = sum(Fsol_Results)
    Ftan_Results = Ftan_Pool.map(p_Ftan, ele_list)
    nrFtan = sum(Ftan_Results)

    return nrFunc, nrFtan

def newton_raph(xn, nodes, np_n, np_e, n_ele, el_type, el_order, order, con_type, c_vals, num_pro, iters, tol):
    nrFtanSolOld = np.zeros((int(len(np_n[:, 0]))*3, int(len(np_n[:, 0]))*3))
    diff = np.zeros(int(len(np_n[:, 0]))*3)
    for i in range(0, iters, 1):
        nrFunc, nrFtan = nonlinear_solve(xn, np_n, np_e, con_type, c_vals, n_ele, num_pro, el_type, el_order, order)
        nrFtanSol = np.copy(nrFtan)
        if nodes != None:
            nrFunc[nodes] = 0
            for idx in nodes:
                nrFtanSol[idx, :] = 0
                nrFtanSol[:, idx] = 0
                nrFtanSol[idx, idx] = nrFtan[idx, idx]
        # print(nrFtan, nrFunc)
        newtStep = np.matmul(np.linalg.inv(nrFtanSol), nrFunc)
        xn1 = xn - newtStep
        # diff = np.c_[diff, (nrFtanSol - nrFtanSolOld).mean(axis=1)]
        # j = diff.shape[1]  # Get the number of columns in 'diff'
        # for row in range(0, diff.shape[0], 1):
        #     plt.plot(list(range(0, j)), diff[row, :], label=f'Row {row + 1}')
        
        print("Residual Average: {}".format(np.average(nrFunc)))
        print("Iteration Number: {}".format(i))

        if abs(np.average(nrFunc)) < tol:
            return xn1, i
        # plt.plot(xn)
        # plt.show()
        xn = xn1
        nrFtanSolOld = nrFtanSol
    # plt.show()
    print("Did not converge")
    return xn, iters

def plot_disps(np_n, np_e, u, n_ele, el_type, el_order, order):

    cmap = get_cmap('Blues')

    dim, n_el_n, xez, phi, _ = element_assign(el_type, el_order)
    _, gp = gauss_num_int(el_type, order)
    x_gp = np.zeros(n_ele*len(gp))
    y_gp = np.zeros(n_ele*len(gp))
    z_gp = np.zeros(n_ele*len(gp))
    u_gp = np.zeros(n_ele*len(gp))
    v_gp = np.zeros(n_ele*len(gp))
    w_gp = np.zeros(n_ele*len(gp))

    # Store disp vals
    uvw = sym.zeros(n_el_n, dim)
    xyz = sym.zeros(n_el_n, dim)
                       
    for e in range(0, n_ele, 1):
        # Set rows and columns GLOBAL
        rc = np_e[e, :]

        # Find coordinates and displacements
        for i, n in enumerate(rc):
            uvw[i, 0] = u[(rc[i]-1)*dim]
            uvw[i, 1] = u[(rc[i]-1)*dim+1]
            uvw[i, 2] = u[(rc[i]-1)*dim+2]
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