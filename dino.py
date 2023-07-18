import numpy as np
import sympy as sym
import multiprocessing as mp
from functools import partial

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

    _, _, xez, _, delPhi = element_assign(el_type, el_order)
    _, gp = gauss_num_int(el_type, order)

    Fdef = sym.zeros(3, 3)
    x = sym.Matrix(x)
    X = sym.Matrix(X)
    
    # Determine δ{xy}/δ{xieta} Jacobians and inverse
    jac_xyz = delPhi * x
    # Determine δ{uv}/δ{xieta} Jacboain and δ{uv}/δ{xy}
    jac_XYZ = delPhi * X
    inv_jac_XYZ = jac_XYZ.inv()
    dxyzdXYZ = inv_jac_XYZ * jac_xyz

    for p in gp:
        Fdef += dxyzdXYZ.subs({xez[0]: p[0], xez[1]: p[1], xez[2]: p[2]})

    Fdef = np.array(Fdef).astype(float)
    Fdef = Fdef/order
    detF = np.linalg.det(Fdef)

    # print(Fdef, x, X)
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

    Fdef, _ = deformation_gradient(X_ele, x_ele, el_type, el_order, order)

    # Jacobian
    jac = delPhi * X_ele
    detJ = jac.det()
    b_sub = jac.inv() * delPhi

    # B Matrix
    b_mat = sym.zeros(6,dim*n_el_n)
    for r, c in enumerate(range(0, dim*n_el_n, dim)):
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

def constitutive_eqs(e, con_type, c_vals, np_n, np_e, x, el_type, el_order, order):

    dim, n_el_n, _, _, _ = element_assign(el_type, el_order)

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

    Fdef, detF = deformation_gradient(X_ele, x_ele, el_type, el_order, order)
    Cgre = Fdef.T * Fdef

    for r in range(0, n_el_n, 1):
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

def geometric_tangent_k(e, Spk, np_e, k_geo, el_type, el_order):

    dim, n_el_n, _, _, delPhi = element_assign(el_type, el_order)

    Spk = Spk[e]
    rc = np_e[e, :]
    ij = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [0,2]])
    for r, n in enumerate(rc):
        Idn = sym.eye(3)
        Geo = sym.zeros(n_el_n, n_el_n)
        Kgeo = sym.zeros(3, 3)
        for a in range(0, n_el_n, 1):
            for b in range(0, n_el_n, 1):
                Geo[a, b] = delPhi[ij[0,0], a] * float(Spk[ij[0,0], ij[0,1]]) * delPhi[ij[0,1], b] + \
                            delPhi[ij[1,0], a] * float(Spk[ij[1,0], ij[1,1]]) * delPhi[ij[1,1], b] + \
                            delPhi[ij[2,0], a] * float(Spk[ij[2,0], ij[1,1]]) * delPhi[ij[2,1], b] + \
                            delPhi[ij[3,0], a] * float(Spk[ij[3,0], ij[3,1]]) * delPhi[ij[3,1], b] + \
                            delPhi[ij[4,0], a] * float(Spk[ij[4,0], ij[4,1]]) * delPhi[ij[4,1], b] + \
                            delPhi[ij[5,0], a] * float(Spk[ij[5,0], ij[5,1]]) * delPhi[ij[5,1], b]
                Kgeo = Kgeo + Geo[a, b] * Idn
        k_geo[dim*(n-1):dim*(n-1)+3, dim*(n-1):dim*(n-1)+3] = k_geo[dim*(n-1):dim*(n-1)+3, dim*(n-1):dim*(n-1)+3] + Kgeo
    return k_geo
    
def gauss_int_fsol(e, Bmat, detJ, Smat, k_sol, np_e, el_type, el_order, order):
     
    dim, _, xez, _, _ = element_assign(el_type, el_order)
    we, gp = gauss_num_int(el_type, order)

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
        for q, w in enumerate(we):
            # [1 2 3]
            term_sol[dim*i+0] = term_sol[dim*i+0] + w * \
                                 float((term[0]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term_sol[dim*i+1] = term_sol[dim*i+1] + w * \
                                 float((term[1]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term_sol[dim*i+2] = term_sol[dim*i+2] + w * \
                                 float((term[2]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))

        k_sol[dim*(n-1)+0] = k_sol[dim*(n-1)+0] + term_sol[dim*r+0] 
        k_sol[dim*(n-1)+1] = k_sol[dim*(n-1)+1] + term_sol[dim*r+1] 
        k_sol[dim*(n-1)+2] = k_sol[dim*(n-1)+2] + term_sol[dim*r+2] 

    return k_sol

def gauss_int_ftan(e, Bmat, detJ, Dmat, Gmat, k_tan, np_e, el_type, el_order, order):

    dim, _, xez, _, _ = element_assign(el_type, el_order)
    we, gp = gauss_num_int(el_type, order)

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
        for q, w in enumerate(we):
            # [1 2 3]
            term_tan[dim*i+0, dim*i+0] = term_tan[dim*i+0, dim*i+0] + w * \
                                          float((term[0, 0]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term_tan[dim*i+0, dim*i+1] = term_tan[dim*i+0, dim*i+1] + w * \
                                          float((term[0, 1]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term_tan[dim*i+0, dim*i+2] = term_tan[dim*i+0, dim*i+2] + w * \
                                          float((term[0, 2]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            # [4 5 6]
            term_tan[dim*i+1, dim*i+0] = term_tan[dim*i+1, dim*i+0] + w * \
                                          float((term[1, 0]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term_tan[dim*i+1, dim*i+1] = term_tan[dim*i+1, dim*i+1] + w * \
                                          float((term[1, 1]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term_tan[dim*i+1, dim*i+2] = term_tan[dim*i+1, dim*i+2] + w * \
                                          float((term[1, 2]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            # [4 5 6]
            term_tan[dim*i+2, dim*i+0] = term_tan[dim*i+2, dim*i+0] + w * \
                                          float((term[2, 0]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term_tan[dim*i+2, dim*i+1] = term_tan[dim*i+2, dim*i+1] + w * \
                                          float((term[2, 1]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))
            term_tan[dim*i+2, dim*i+2] = term_tan[dim*i+2, dim*i+2] + w * \
                                          float((term[2, 2]*detJ).subs({xez[0]: gp[q, 0], xez[1]: gp[q, 1], xez[2]: gp[q, 2]}))

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

def nonlinear_solve(u, np_n, np_e, con_type, c_vals, n_ele, num_pro, el_type, el_order, order):

    dim, _, _, _, _ = element_assign(el_type, el_order)

    n_n = int(len(np_n[:, 0]))
    x = np_n[:, 1:].flatten() 
    x = x + u

    k_sol = np.zeros(dim*n_n)
    k_tan = np.zeros((dim*n_n, dim*n_n))
    k_geo = sym.zeros(dim*n_n, dim*n_n)

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
                     el_type=el_type, el_order=el_order)
    Kgeos = Gmat_Pool.map(p_Gmat, ele_list)

    p_Fsol = partial(gauss_int_fsol, Bmat=Bmats, detJ=detJs, Smat=Smats, \
                        k_sol=k_sol, np_e=np_e, el_type=el_type, el_order=el_order, order=order)
    
    p_Ftan = partial(gauss_int_ftan, Bmat=Bmats, detJ=detJs, Dmat=Dmats, Gmat=Kgeos, \
                        k_tan=k_tan, np_e=np_e, el_type=el_type, el_order=el_order, order=order)
    
    Fsol_Results = Fsol_Pool.map(p_Fsol, ele_list)
    nrFunc = sum(Fsol_Results)
    Ftan_Results = Ftan_Pool.map(p_Ftan, ele_list)
    nrFtan = sum(Ftan_Results)

    return nrFunc, nrFtan

def newton_raph(u, nodes, np_n, np_e, n_ele, el_type, el_order, order, con_type, c_vals, num_pro, iters, tol):
    for i in range(0, iters, 1):
        nrFunc, nrFtan = nonlinear_solve(u, np_n, np_e, con_type, c_vals, n_ele, num_pro, el_type, el_order, order)
        newtStep = np.matmul(np.linalg.inv(nrFtan), nrFunc)
        newtStep[nodes] = 0
        da = u - newtStep
        diff = np.average(da - u)
        print("Residual Average: {}".format(np.average(nrFunc)))
        print("Iteration Number: {}".format(i))
        if abs(diff) < tol:
            return da, i
        u = da
    
    print("Did not converge")
    return u, iters
