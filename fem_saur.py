"""
FILE: fem.py
AUTHOR: Liam Murray (murrayla@student.unimelb.edu.au)
DESCRIPTION: Generalised tool for solving PDEs with the Finite Element Method
"""

# Library Dependencies
import numpy as np
import multiprocessing as mp

# File Dependencies
import saur

# Global Variables
DIRECTORY = "GitHub/Dino/"
FILE_NAME = "oneTetTest"
CONSTITUTIVE_TYPE = 0
C_VALS = [0.092, 0.237]
E_MOD = 200 
NU = 0.20
NUM_PROCESSES = 4
ITERATIONS = 10
TOLERANCE = 1.48e-8

def main():

    ## SETUP START ##
    # --

    # Intake Mesh
    # dino.nodes_and_elements(DIRECTORY + "gmsh_" + FILE_NAME + ".msh", type_num=11)
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

    # --
    ## SETUP END ##
    
    ## BEGIN NEWTON RAPHSON ##
    ## -- 

    dim = 3
    u = np.zeros(n_n*dim)
    # nodes = None
    nodes = list()
    u, nodes = saur.apply_nonlinear_BC(np_n, u, nodes, BC0=[0, 0, 0], BC1=[3, 0, 0], axi=0)
    u, nodes = saur.apply_nonlinear_BC(np_n, u, nodes, BC0=[None, 0, None], BC1=[None, None, None], axi=1)
    u, nodes = saur.apply_nonlinear_BC(np_n, u, nodes, BC0=[None, None, None], BC1=[None, None, None], axi=2)
    
    root, it = saur.newton_raph(u, nodes, np_n, np_e, n_ele, C_VALS, NUM_PROCESSES, ITERATIONS, TOLERANCE)
    
    ## -- 
    ## END NEWTON RAPHSON ##

    print("After {} iterations we have:".format(it))
    print(root)

    # dino.plot_geo(np_n, np_e, root)
    saur.plot_disps(np_n, np_e, root, n_ele)

if __name__ == '__main__':
    mp.freeze_support()
    main()