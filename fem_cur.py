"""
FILE: fem.py
AUTHOR: Liam Murray (murrayla@student.unimelb.edu.au)
DESCRIPTION: Generalised tool for solving PDEs with the Finite Element Method
"""

# Library Dependencies
import numpy as np
import sympy as sym
import multiprocessing as mp

# File Dependencies
from saur_analytic import *

# Global Variables
DIRECTORY = "GitHub/Dino/"
FILE_NAME = "oneTetTest"
CONSTITUTIVE_TYPE = 0
C_VALS = [0.092, 0.237]
E_MOD = 200 
NU = 0.20
NUM_PROCESSES = 4
ITERATIONS = 5
TOLERANCE = 1.48e-8
XI = sym.Symbol('XI', real=True)
ETA = sym.Symbol('ETA', real=True)
ZETA = sym.Symbol('ZETA', real=True)
beta = 1 - XI - ETA - ZETA # Dependent

def main():

    ## SETUP START ##
    # --

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
    np_n = np.array(n_list[1:]).astype(float)
    np_e = np.array(e_list[1:])
    np_e = np_e[:, 3:].astype(int)

    # Determine Parameters
    n_ele = len(np_e[:, 0])
    n_n = int(len(np_n[:, 0]))

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
    # Shape Functions
    phi = sym.Matrix([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10])
    # Derivative of Shape Functions
    # dNdxez = [δφ1/δξ δφ2/δξ ... δφ10/δξ
    #           δφ1/δη δφ2/δη ... δφ10/δη
    #           δφ1/δζ δφ2/δζ ... δφ10/δζ]
    dNdxez = sym.Matrix(
        [
            [sym.diff(phi[j], XI, 1) for j in range(0, N_EL_N, 1)],  
            [sym.diff(phi[k], ETA, 1) for k in range(0, N_EL_N, 1)],
            [sym.diff(phi[m], ZETA, 1) for m in range(0, N_EL_N, 1)]
        ]
    )

    # --
    ## SETUP END ##
    
    ## BEGIN NEWTON RAPHSON ##
    ## -- 

    dim = 3
    u = np.zeros(n_n*dim)
    nodes = None
    # nodes = list()
    # u, nodes = apply_nonlinear_BC(np_n, u, nodes, BC0=[0, 0, 0], BC1=[3, 0, 0], axi=0)
    # u, nodes = apply_nonlinear_BC(np_n, u, nodes, BC0=[None, 0, None], BC1=[None, None, None], axi=1)
    # u, nodes = apply_nonlinear_BC(np_n, u, nodes, BC0=[None, None, None], BC1=[None, None, None], axi=2)
    
    root, it = newton_raph(u, nodes, np_n, np_e, n_ele, dNdxez, C_VALS, NUM_PROCESSES, ITERATIONS, TOLERANCE)
    
    ## -- 
    ## END NEWTON RAPHSON ##

    print("After {} iterations we have:".format(it))
    print(root)

    # dino.plot_geo(np_n, np_e, root)
    plot_disps(np_n, np_e, root, n_ele)

if __name__ == '__main__':
    mp.freeze_support()
    main()