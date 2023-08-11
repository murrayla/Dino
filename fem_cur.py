"""
FILE: fem.py
AUTHOR: Liam Murray (murrayla@student.unimelb.edu.au)
DESCRIPTION: Generalised tool for solving PDEs with the Finite Element Method
"""

# Library Dependencies
import numpy as np
import multiprocessing as mp

# File Dependencies
from dino_cur import *

# Global Variables
DIRECTORY = "GitHub/Dino/"
FILE_NAME = "cubeTest"
CONSTITUTIVE_TYPE = 0
C_VALS = [1, 1] #0.092,0.237]
E_MOD = 200 
NU = 0.20
NUM_PROCESSES = 4
ITERATIONS = 10
TOLERANCE = 1e-8
GP = np.array(
    [
        [1/4, 1/4, 1/4], 
        [1/2, 1/6, 1/6], 
        [1/6, 1/2, 1/6], 
        [1/6, 1/6, 1/2],
        [1/6, 1/6, 1/6]
    ]
)

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
ORDER = len(WE)

""" 
[Note: β = 1 - ξ - η - ζ]
[
    φ0 = ξ*(ξ-1), φ1 = η*(η-1), φ2 = ζ*(ζ-1), φ3 = β*(β-1)
    φ4 = 4*ξ*η, φ5 = 4*η*ζ, φ6 = 4*ζ*ξ, φ7 = 4*ξ*β
    φ8 = 4*ζ*β, φ9 = 4*β*η
] @ Gauss
"""
PHI = np.zeros((ORDER, N_EL_N))

"""
[
    δφ1/δξ δφ2/δξ ... δφ10/δξ
    δφ1/δη δφ2/δη ... δφ10/δη
    δφ1/δζ δφ2/δζ ... δφ10/δζ
] @ Gauss
"""
DEL_PHI = np.zeros((ORDER, DIM, N_EL_N))

for x in range(0, ORDER, 1):
    # Shape Functions
    PHI[x, :] = np.array(
        [
            GP[x, 0]*(2*GP[x, 0]-1),        
            GP[x, 1]*(2*GP[x, 1]-1),
            GP[x, 2]*(2*GP[x, 2]-1),
            (1 - GP[x, 0] - GP[x, 1] - GP[x, 2])*(2*(1 - GP[x, 0] - GP[x, 1] - GP[x, 2])-1),
            4*GP[x, 0]*GP[x, 1],
            4*GP[x, 1]*GP[x, 2],
            4*GP[x, 2]*GP[x, 0],
            4*GP[x, 0]*(1 - GP[x, 0] - GP[x, 1] - GP[x, 2]),
            4*GP[x, 2]*(1 - GP[x, 0] - GP[x, 1] - GP[x, 2]),
            4*GP[x, 1]*(1 - GP[x, 0] - GP[x, 1] - GP[x, 2])
        ]
    )
    # Derivatives of shape functions
    DEL_PHI[x, :, :] = np.array(
        [
            [4*GP[x, 0] - 1, 0, 0, 4*GP[x, 1] + 4*GP[x, 0] + 4*GP[x, 2] - 3, 4*GP[x, 1], 0, 4*GP[x, 2], -4*GP[x, 1] - 8*GP[x, 0] - 4*GP[x, 2] + 4, -4*GP[x, 2], -4*GP[x, 1]], 
            [0, 4*GP[x, 1] - 1, 0, 4*GP[x, 1] + 4*GP[x, 0] + 4*GP[x, 2] - 3, 4*GP[x, 0], 4*GP[x, 2], 0, -4*GP[x, 0], -4*GP[x, 2], -8*GP[x, 1] - 4*GP[x, 0] - 4*GP[x, 2] + 4], 
            [0, 0, 4*GP[x, 2] - 1, 4*GP[x, 1] + 4*GP[x, 0] + 4*GP[x, 2] - 3, 0, 4*GP[x, 1], 4*GP[x, 0], -4*GP[x, 0], -4*GP[x, 1] - 4*GP[x, 0] - 8*GP[x, 2] + 4, -4*GP[x, 1]]
        ]
    )

def main():

    # ============================== #
    # Setup
    # ============================== #

    # Intake Mesh
    # nodes_and_elements(DIRECTORY + "gmsh_" + FILE_NAME + ".msh", type_num=11)
    nodes = open(DIRECTORY + FILE_NAME + "_cvt2dino.nodes", 'r')
    elems = open(DIRECTORY + FILE_NAME + "_cvt2dino.ele", 'r')
    n_list = list()
    e_list = list()

    # Store Node and Element Data
    for line in nodes:
        n_list.append(line.strip().replace('\t', ' ').split(' '))
    for line in elems:
        e_list.append(line.strip().replace('\t', ' ').split(' '))
    np_n = np.array(n_list[1:]).astype(np.float64)
    np_e = np.array(e_list[1:])
    np_e = np_e[:, 3:].astype(np.int32)

    # Determine Parameters
    n_ele = len(np_e[:, 0])
    n_n = int(len(np_n[:, 0]))

    # ============================== #
    # Solver
    # ============================== #

    dim = 3
    u = np.zeros(n_n*dim)
    # u = np.array(
    #     [0,         0,          0,          0,          0,          0,
    #         0,          1,          0,          0,          0,          0,
    #         0.17297911,  0,          0,         -0.49423616, -0.0076786,   0,
    #         0,         -0.1717379,   0,         0,          0,          0.17297911,
    #         0.3552232,   0,          0.3552232,   0,         -0.0076786,  -0.49423616
    #     ]
    # )
    # nodes = None
    nodes = list()
    u, nodes = apply_nonlinear_BC(np_n, u, nodes, BC0=[0, None, None], BC1=[1, None, None], axi=0)
    u, nodes = apply_nonlinear_BC(np_n, u, nodes, BC0=[None, 0, None], BC1=[None, 0, None], axi=1)
    u, nodes = apply_nonlinear_BC(np_n, u, nodes, BC0=[None, None, 0], BC1=[None, None, None], axi=2)
    
    root, it = newton_raph(u, nodes, np_n, np_e, n_ele, DEL_PHI, C_VALS, NUM_PROCESSES, ITERATIONS, TOLERANCE)
    
    # ============================== #
    # Display
    # ============================== #

    print("After {} iterations we have:".format(it))
    print(root)

    # dino.plot_geo(np_n, np_e, root)
    plot_disps(np_n, np_e, root, n_ele, PHI)

if __name__ == '__main__':
    mp.freeze_support()
    main()