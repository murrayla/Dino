import numpy as np
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from functools import partial

# data = [
#     [0.9197896733368800, 0.0267367755543735, 0.0267367755543735, 0.0021900463965388],
#     [0.0267367755543735, 0.9197896733368800, 0.0267367755543735, 0.0021900463965388],
#     [0.0267367755543735, 0.0267367755543735, 0.9197896733368800, 0.0021900463965388],
#     [0.0267367755543735, 0.0267367755543735, 0.0267367755543735, 0.0021900463965388],
#     [0.1740356302468940, 0.7477598884818090, 0.0391022406356488, 0.0143395670177665],
#     [0.7477598884818090, 0.1740356302468940, 0.0391022406356488, 0.0143395670177665],
#     [0.1740356302468940, 0.0391022406356488, 0.7477598884818090, 0.0143395670177665],
#     [0.7477598884818090, 0.0391022406356488, 0.1740356302468940, 0.0143395670177665],
#     [0.1740356302468940, 0.0391022406356488, 0.0391022406356488, 0.0143395670177665],
#     [0.7477598884818090, 0.0391022406356488, 0.0391022406356488, 0.0143395670177665],
#     [0.0391022406356488, 0.1740356302468940, 0.7477598884818090, 0.0143395670177665],
#     [0.0391022406356488, 0.7477598884818090, 0.1740356302468940, 0.0143395670177665],
#     [0.0391022406356488, 0.1740356302468940, 0.0391022406356488, 0.0143395670177665],
#     [0.0391022406356488, 0.7477598884818090, 0.0391022406356488, 0.0143395670177665],
#     [0.0391022406356488, 0.0391022406356488, 0.1740356302468940, 0.0143395670177665],
#     [0.0391022406356488, 0.0391022406356488, 0.7477598884818090, 0.0143395670177665],
#     [0.4547545999844830, 0.4547545999844830, 0.0452454000155172, 0.0250305395686746],
#     [0.4547545999844830, 0.0452454000155172, 0.4547545999844830, 0.0250305395686746],
#     [0.4547545999844830, 0.0452454000155172, 0.0452454000155172, 0.0250305395686746],
#     [0.0452454000155172, 0.4547545999844830, 0.4547545999844830, 0.0250305395686746],
#     [0.0452454000155172, 0.4547545999844830, 0.0452454000155172, 0.0250305395686746],
#     [0.0452454000155172, 0.0452454000155172, 0.4547545999844830, 0.0250305395686746],
#     [0.5031186450145980, 0.2232010379623150, 0.2232010379623150, 0.0479839333057554],
#     [0.2232010379623150, 0.5031186450145980, 0.2232010379623150, 0.0479839333057554],
#     [0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.0479839333057554],
#     [0.5031186450145980, 0.2232010379623150, 0.0504792790607720, 0.0479839333057554],
#     [0.2232010379623150, 0.5031186450145980, 0.0504792790607720, 0.0479839333057554],
#     [0.2232010379623150, 0.2232010379623150, 0.0504792790607720, 0.0479839333057554],
#     [0.5031186450145980, 0.0504792790607720, 0.2232010379623150, 0.0479839333057554],
#     [0.2232010379623150, 0.0504792790607720, 0.5031186450145980, 0.0479839333057554],
#     [0.2232010379623150, 0.0504792790607720, 0.2232010379623150, 0.0479839333057554],
#     [0.0504792790607720, 0.5031186450145980, 0.2232010379623150, 0.0479839333057554],
#     [0.0504792790607720, 0.2232010379623150, 0.5031186450145980, 0.0479839333057554],
#     [0.0504792790607720, 0.2232010379623150, 0.2232010379623150, 0.0479839333057554],
#     [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.0931745731195340]
# ]
# GP = np.array(data)[:, :3]
# WE = np.array(data)[:, 3]

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

ORDER = len(WE)
DIM = 3
N_EL_N = 10
I = np.eye(DIM)
IJ = np.array([[0,0], [1,1], [2,2], [0,1], [1,2], [2,0]])
TMAP = {
        (0,0): 0, (1,1): 1, (2,2): 2, 
        (0,1): 3, (1,0): 3, (1,2): 4, 
        (2,1): 4, (2,0): 5, (0,2): 5
}

def dirichlet(np_n, bcs):
    # Initialize arrays to store displacement values and node numbers with BCs
    u = np.zeros((np_n.shape[0], np_n.shape[1] - 1))
    nodes = []

    # Define a helper function to apply BCs and track BC nodes
    def apply_bc(condition, bc_values, dim_idx):
        nonlocal nodes
        nonlocal u
        u[condition, dim_idx] = bc_values
        nodes.extend(np.where(condition)[0] * DIM + dim_idx)

    # Apply BCs
    for idx, axi in enumerate(['X', 'Y', 'Z']):

        min_bc = bcs['min'][axi]
        max_bc = bcs['max'][axi]

        condition_min = np_n[:, idx + 1] == np.min(np_n[:, idx + 1])
        condition_max = np_n[:, idx + 1] == np.max(np_n[:, idx + 1])

        for dim_idx in range(0, DIM, 1):

            if min_bc[dim_idx] is not None:
                u[condition_min, dim_idx] = min_bc[dim_idx]
                nodes.extend(np.where(condition_min)[0] * DIM + dim_idx)

            if max_bc[dim_idx] is not None:
                u[condition_max, dim_idx] = max_bc[dim_idx]
                nodes.extend(np.where(condition_max)[0] * DIM + dim_idx)

    # # Apply BCs at the center of X, Y, and Z
    # center_bc = bcs['center']
    # for dim_idx, bc_value in enumerate(center_bc):
    #     condition = np_n[:, dim_idx + 1] == bc_value
    #     apply_bc(condition, bc_value, dim_idx)

    # # Remove duplicates from the list of BC nodes
    # nodes = list(set(nodes))

    # for row, xyz in enumerate(u):
    #     if np.isnan(xyz[0]):
    #         nodes.remove(row*DIM + 0)
    #     if np.isnan(xyz[1]):
    #         nodes.remove(row*DIM + 1)
    #     if np.isnan(xyz[2]):
    #         nodes.remove(row*DIM + 2)

    u = u.flatten()
    # u[np.isnan(u)] = 0

    return u, np.array(nodes)

def neumann(np_n, bcs):
    rhs = np.zeros((np_n.shape[0], np_n.shape[1] - 1)).flatten()

    for i, (x, y, z) in enumerate(np_n[:, 1:]):
        # if x == 100:
        #     rhs[DIM*i] = 0.1
        vecNorm = np.linalg.norm([x, y])
        if abs(vecNorm - bcs['pos']) < 1e-3 and abs(z - 1) < 1e-3:
            rhs[DIM*i + 0] = x / vecNorm * bcs['val']
            rhs[DIM*i + 1] = y / vecNorm * bcs['val']
    return rhs

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

def cur_B_mat(e, np_n, np_e, x, dN):

    # Preallocate
    cur = np.zeros((N_EL_N, DIM))
    bmat = np.zeros((ORDER, DIM*2, N_EL_N*DIM))

    # Number of nodes and deformed x
    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)

    # Relevant node numbers for element
    for i, local_node in enumerate(np_e[e, :]):
        cur[i, :] = xc[np.where(np_n[:, 0] == local_node), :][0]

    # ============================== #
    # Create B Matrix
    # ============================== #

    # Loop through each column index c
    for q in range(0, ORDER, 1):
        # dNdxyz = [∂φ1/∂x ∂φ2/∂x ... ∂φ10/∂x
        #           ∂φ1/∂y ∂φ2/∂y ... ∂φ10/∂y
        #           ∂φ1/∂z ∂φ2/∂z ... ∂φ10/∂z] @ Gauss   
        dNdxyz = np.matmul(
            np.linalg.inv(
                np.matmul(dN[q, :, :], cur)
            ), dN[q, :, :]
        )
        for r, c in enumerate(range(0, DIM*N_EL_N, DIM)):
            # [∂φ/∂x,      0,      0] 
            # [    0,  ∂φ/∂y,      0]
            # [    0,      0,  ∂φ/∂z]
            # [∂φ/∂y,  ∂φ/∂x,      0]
            # [    0,  ∂φ/∂z,  ∂φ/∂y]
            # [∂φ/∂z,      0,  ∂φ/∂x]
            bmat[q, :, c:c+3] = np.array(
                [
                    [dNdxyz[0, r], 0, 0],
                    [0, dNdxyz[1, r], 0],
                    [0, 0, dNdxyz[2, r]],
                    [dNdxyz[1, r], dNdxyz[0, r], 0],
                    [0, dNdxyz[2, r], dNdxyz[1, r]],
                    [dNdxyz[2, r], 0, dNdxyz[0, r]]
                ]
            )

    return bmat

def constitutive_eqs(e, c_vals, np_n, np_e, x, dN):

    # Number of nodes and deformed x
    n_n = int(len(np_n[:, 0]))
    xc = x.reshape(n_n, DIM)
    
    # Preallocate 
    cur = np.zeros((N_EL_N, DIM))
    ref = np.zeros((N_EL_N, DIM))
    cau = np.zeros((ORDER, DIM, DIM))
    ela = np.zeros((ORDER, DIM*2, DIM*2))
    dgr = np.zeros((ORDER, DIM, DIM))
    jac = np.zeros(ORDER)
    

    # Relevant node numebrs for element
    rc = np_e[e, :]
    for i, local_node in enumerate(rc):
        cur[i, :] = xc[np.where(np_n[:, 0] == local_node), :][0]
        ref[i, :] = np_n[np.where(np_n[:, 0] == local_node), 1:][0]

    # ============================== #
    # Determine Deformation Gradient
    # ============================== #

    # Sub Gauss
    for q in range(0, ORDER, 1):
        # F = [∂x/∂X ∂x/∂Y ∂x/∂Z
        #      ∂y/∂X ∂y/∂Y ∂y/∂Z
        #      ∂z/∂X ∂z/∂Y ∂z/∂Z] @ Gauss
        dgr[q, :, :] = np.matmul(
            np.matmul(
                np.linalg.inv(
                    np.matmul(dN[q, :, :], ref)
                ), dN[q, :, :]
            ), cur
        )
        # fdet = |F| @ Gauss
        jac[q] = np.linalg.det(dgr[q, :, :])

    # ============================== #
    # Find Elastic Moduli, D
    # Determine Sigma, σ
    # ============================== #

    # Nearly incompressible
    d = 1000.1**-1
    Em   = 200 
    nu  = 0.20
    mu  = Em/(2*(1+nu))
    lam = nu*Em/((1+nu)*(1-2*nu))

    for n in range(0, ORDER, 1):
        # b = F*F^T @ Gauss
        f = dgr[n, :, :]
        b = np.matmul(f, np.transpose(f))
        c = np.matmul(np.transpose(f), f)
        invC = np.linalg.inv(c)
        trb = np.trace(b)
        trc = np.trace(c)

        ## ==== Mooney Rivlin ==== #
        # # W(I1, I2, J) = c1 * (I1 - 3) + c2 * (I2 - 3) + 1/d * (J - 1)^2
        # # Derivatives of Energy in terms of Invariants
        # dWdI = np.array([c_vals[0], c_vals[1], 2/d*(fdet[n]-1)])
        # ddWdII = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2/d]])

        ## ==== neo-Hookean ==== #
        # W = 0.5 * mu * (I1 - 3 - 2 * ln(J)) + 0.5 * lambda * (J - 1)^2
        # Derivatives of Energy in terms of Invariants
        dWdI = np.array([0.5*mu, 0, lam*(jac[n]-1) - mu*1/jac[n]])
        ddWdII = np.array([[0, 0, 0], [0, 0, 0], [0, 0, mu * 1/(jac[n]**2) + lam]])

        ## ==== Cauchy Stress === #
        # σ = [σ11, σ22, σ33, σ12, σ23, σ31]
        # Fill cauchy stress
        sPK = np.zeros((DIM, DIM))

        ## ==== Elastic Moduli === #
        d = np.zeros((DIM*2,DIM*2))

        # [4 * ∂W/∂II, ∂W/∂J]
        term4 = np.array(
            [
                [4*dWdI[1]], 
                [dWdI[2]]
            ]
        )
        for i in range(0, DIM, 1):
            for j in range(0, DIM, 1):
                sPK[i, j] = 2 * (
                    dWdI[0] * I[i, j] + 
                    dWdI[1] * (trc * I[i, j] - c[i, j]) + 
                    dWdI[2] * 0.5 * jac[n] * invC[i, j]
                )
                # [bij, (I * bij - bim * bmj), 0.5 * J * δij]
                term1 = np.array(
                    [
                        [
                            b[i, j], 
                            trb * b[i, j] - (b[i, 0]*b[0, j] + b[i, 1]*b[1, j] + b[i, 2]*b[2, j]), 
                            0.5 * jac[n] * I[i, j]
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
                                [0.5 * jac[n] * I[k, l]]
                            ]
                        )
                        # 0.5 * [δik * δjl + δil * δjk]
                        lil_b = 0.5 * (I[i, k] * I[j, l] + I[i, l] * I[j, k])
                        # [bij * bkl - 0.5 * (bik * bjl + bil * bjk), J * (δkl * δij - 2 * lil_b)]
                        term3 = np.array(
                            [
                                [
                                    b[i, j] * b[k, l] - 0.5 * (b[i, k] * b[j, l] + b[i, l] * b[j, k]),
                                    jac[n] * (I[i, j] * I[k, l] - 2*lil_b)
                                ]
                            ]
                        )
                        d[TMAP[(i,j)], TMAP[(k,l)]] += np.matmul(term1, term2) + np.matmul(term3, term4)

        cau[n, :, :] = 1/jac[n] * np.matmul(np.matmul(f, sPK), np.transpose(f))                    
        ela[n, :, :] = d
    
    return cau, ela

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
    Fa = np.zeros((N_EL_N*DIM, 1))

    # ============================== #
    # Integration of Kab, Gab, Fa
    # ============================== #

    for q, w in enumerate(WE):
        # J = [∂x/∂ξ ∂y/∂ξ ∂z/∂ξ
        #      ∂x/∂η ∂y/∂η ∂z/∂η
        #      ∂x/∂ζ ∂y/∂ζ ∂z/∂ζ] @ Gauss
        jac = np.matmul(dN[q, :, :], cur)
        # dNdxyz = [∂φ1/∂x ∂φ2/∂x ... ∂φ10/∂x
        #           ∂φ1/∂y ∂φ2/∂y ... ∂φ10/∂y
        #           ∂φ1/∂z ∂φ2/∂z ... ∂φ10/∂z] @ Gauss                
        dNdxyz = np.matmul(np.linalg.inv(jac), dN[q, :, :])
        # Voigt form of stress
        voigt = np.array([[c[q, 0, 0]], [c[q, 1, 1]], [c[q, 2, 2]], [c[q, 0, 1]], [c[q, 1, 2]], [c[q, 2, 0]]])
        for al in range(N_EL_N):
            for be in range(N_EL_N):
                # Geometric / Initial Stiffness Gαβ = ∫ Nα,i * σij * Nβ,j dv 
                # g = 0
                # for i, j in IJ:
                #     g += dNdxyz[i, al] * c[q, i, j] * dNdxyz[j, be]
                # Gab[al, be] = g * w
                                
                Gab[al, be] += np.matmul(
                    np.transpose(dNdxyz[:, al]), np.matmul(c[q, :, :], dNdxyz[:, be])
                ) * w 
                # Material Stiffness Km = ∫ BαT * DT * Bβ dv
                Kab[DIM*al:DIM*al+DIM, DIM*be:DIM*be+DIM] += np.matmul(
                    np.matmul(
                        np.transpose(b[q, :, DIM*al:DIM*al+DIM]), d[q, :, :]
                    ), b[q, :, DIM*be:DIM*be+DIM]
                ) * w
            # Residual Fα = - ∫ BαTσ dv OR Fα = - ∫ σ * ∂Nα/∂x dv
            Fa[DIM*al:DIM*al+DIM] -= np.matmul(np.transpose(b[q, :, DIM*al:DIM*al+DIM]), voigt) * w

    # ============================== #
    # Array allocation
    # ============================== #

    # Fill Fsol & Ftan
    # Loop rows
    for i in range(0, N_EL_N, 1):
        # Loop columns
        for j in range(0, N_EL_N, 1):
            # Allocate to Tangent
            kT[
                DIM*(rc[i]-1):DIM*(rc[i]-1)+DIM, DIM*(rc[j]-1):DIM*(rc[j]-1)+DIM
            ] += (Kab[DIM*i:DIM*i+DIM, DIM*j:DIM*j+DIM] + Gab[i, j] * I)
        # Allocate to Residual
        Fs[DIM*(rc[i]-1):DIM*(rc[i]-1)+DIM] += Fa[DIM*i:DIM*i+3, 0]

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

def newton_raph(u, dir_n, f, np_n, np_e, n_ele, dN, c_vals, num_pro, iters, tol):

    # ============================== #
    # Newton Raphson Solver
    # ============================== #

    xn = np_n[:, 1:].flatten() + u 
    # return u, 1
    for i in range(0, iters, 1):
        nrF, nrKT = nonlinear_solve(xn, np_n, np_e, dN, c_vals, n_ele, num_pro)

        rhs = f - nrF

        nrKT_sol = np.copy(nrKT)
        rhs_sol = np.copy(rhs)

        if len(dir_n) > 0:
            rhs[dir_n] = 0
            for idx in dir_n:
                nrKT_sol[idx, :] = 0
                nrKT_sol[:, idx] = 0
                nrKT_sol[idx, idx] = nrKT[idx, idx]
            un = sp.linalg.solve(nrKT_sol, rhs)
        else:
            un = sp.linalg.solve(nrKT_sol, rhs)

        xn1 = xn - un

        SSR = sum(np.square(rhs))
        SSU = sum(np.square(un))
        
        xn = xn1

        print("Sum of Squared (RESIDUAL): {}".format(SSR))
        print("Sum of Squared (DELTA): {}".format(SSU))
        print("Iteration Number: {}".format(i))

        if SSR < tol: # or SSU < tol:
            # plt.plot(rhs_sol)
            # plt.show()
            np.savetxt('testCubeForce_2.txt', rhs_sol)
            np.savetxt('testCubeDispl_2.txt', xn)
            np.savetxt('testCubeNodes_2.txt', dir_n)
            return xn - np_n[:, 1:].flatten(), i

    print("Did not converge")
    return xn - np_n[:, 1:].flatten(), iters

def plot_disps(np_n, np_e, u, n_ele, phi):
    plt.plot(u)
    plt.show()
    cmap = get_cmap('seismic')
    # cmap = get_cmap('Blues')

    x_gp = np.zeros(n_ele*len(WE))
    y_gp = np.zeros(n_ele*len(WE))
    z_gp = np.zeros(n_ele*len(WE))
    u_gp = np.zeros(n_ele*len(WE))
    v_gp = np.zeros(n_ele*len(WE))
    w_gp = np.zeros(n_ele*len(WE))

    # Store disp vals
    uvw = np.zeros((N_EL_N, DIM))
    xyz = np.zeros((N_EL_N, DIM))
                       
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