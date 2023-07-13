"""
FILE: fem.py
AUTHOR: Liam Murray (murrayla@student.unimelb.edu.au)
DESCRIPTION: Generalised tool for solving PDEs with the Finite Element Method
"""

# Library Dependencies
import numpy as np
import sympy as sym
import multiprocessing as mp
from functools import partial

# File Dependencies
import dino

# Global Variables
FILE_NAME = "3Dtwotet"
ELEMENT_TYPE = 1
ELEMENT_ORDER = 1
GAUSS_ORDER = 5
E_MOD = 200 
NU = 0.20
NUM_PROCESSES = 6

def main():
    ## SETUP ##
    # Define Interpolation
    dim, n_el_n, sym_vars, phi, delPhi = dino.element_assign(el_type=ELEMENT_ORDER, el_order=ELEMENT_ORDER)
    we, gp = dino.gauss_num_int(el_type=ELEMENT_TYPE, order=GAUSS_ORDER)

    # Intake Mesh
    dino.nodes_and_elements(file_name="msh_files/" + FILE_NAME + ".msh", type_num=11)
    nodes = open("gmsh2iron.nodes", 'r')
    elems = open("gmsh2iron.ele", 'r')

    # Store Node and Element Data
    nodes_list = list()
    elements_list = list()
    for line in nodes:
        nodes_list.append(line.strip().replace('\t', ' ').split(' '))
    for line in elems:
        elements_list.append(line.strip().replace('\t', ' ').split(' '))
    np_nodes  = np.array(nodes_list[1:])
    np_nodes  = np_nodes.astype(float)
    np_eles   = np.array(elements_list[1:])
    np_eles   = np_eles[:, 3:].astype(int)

    # Determine Parameters
    n_ele = len(np_eles[:, 0])
    n_n = int(len(np_nodes[:, 0]))

    # Definite constitutive relationship
    d, mu, lam = dino.get_D_mat(E_MOD, NU)

    ## END SETUP ##
    
    ## BEGIN PARALLELISATION ##

    # # Create a partial function with fixed arguments except ele
    # p_calc_k = partial(dino.calc_k_linear_elasticity, np_nodes=np_nodes, np_eles=np_eles, phi=phi, delPhi=delPhi, \
    #                        d=d, xi=sym_vars[0], eta=sym_vars[1], zeta=sym_vars[2], dim=dim, n_el_n=n_el_n, we=we, gp=gp)

    # Create a partial function with fixed arguments except ele
    p_calc_k = partial(dino.calc_k_laplace, np_nodes=np_nodes, np_eles=np_eles, phi=phi, delPhi=delPhi, \
                           d=d, xi=sym_vars[0], eta=sym_vars[1], zeta=sym_vars[2], dim=dim, n_el_n=n_el_n, we=we, gp=gp)

    # Create a Pool of processes
    first_pool = mp.Pool(processes=NUM_PROCESSES, maxtasksperchild=100)

    # Create a list of ele values to iterate over
    ele_list = range(0, n_ele, 1)

    # Map the partial function to the list of ele values using multiple processes
    p_k_gl = first_pool.map(p_calc_k, ele_list)

    ## END PARALLELISATION ##

    # Matrix Setup
    k_gl = sum(p_k_gl)
    f_gl = np.zeros(dim*n_n)
    k_gl_sol = np.copy(k_gl)
    f_gl_sol = np.copy(f_gl)
    u_sol    = np.zeros(dim*n_n)

    ## BEGIN BOUNDARY CONDITIONS ## 

    # Displacement
    per_disp = 5
    disp     = per_disp/100*(np.amax(np_nodes[:, 3]) - np.amin(np_nodes[:, 3]))

    x0_BC = 0
    y0_BC = 0
    z0_BC = 0

    x1_BC = disp
    y1_BC = 0
    z1_BC = 0

    # # Boundary Conditions
    # k_gl_sol, f_gl_sol = dino.apply_BC_3D(np_nodes, k_gl_sol, f_gl_sol, x0_BC, x1_BC, 0)
    # k_gl_sol, f_gl_sol = dino.apply_BC_3D(np_nodes, k_gl_sol, f_gl_sol, y0_BC, y1_BC, 1)
    # k_gl_sol, f_gl_sol = dino.apply_BC_3D(np_nodes, k_gl_sol, f_gl_sol, z0_BC, z1_BC, 2)

    ## END BOUNDARY CONDITIONS ## 
    f_gl_sol[0] = 10
    f_gl_sol[-1] = -10

    ## Displacements
    rhs = f_gl_sol #- np.matmul(k_gl, u_sol)
    u   = np.matmul(np.linalg.inv(k_gl_sol), rhs)
    f   = np.matmul(k_gl, u)

    ## BEGIN PARALLELISATION ## 

    # Create a partial function with fixed arguments except ele
    p_epsSigU = partial(dino.epsSigU, dim=dim, xi=sym_vars[0], eta=sym_vars[1], zeta=sym_vars[2], np_eles=np_eles, \
                        np_nodes=np_nodes, gp=gp, u=u, phi=phi, delPhi=delPhi, mu=mu, lam=lam)

    # Create a Pool of processes
    second_pool = mp.Pool(processes=NUM_PROCESSES, maxtasksperchild=1000)

    # Map the partial function to the list of ele values using multiple processes
    analysis_results = second_pool.map(p_epsSigU, ele_list)

    # Unpack the results appropriately
    p_stress, p_strain, p_displa, p_coords, p_undefc = zip(*analysis_results)

    ## END PARALLELISATION ## 

    stress = sum(p_stress)
    strain = sum(p_strain)
    displa = sum(p_displa)
    coords = sum(p_coords)
    undefc = sum(p_undefc)

    ## Save Data
    np.savetxt("output_files/" + "k_" + FILE_NAME + ".text", k_gl, fmt='%f')
    np.savetxt("output_files/" + "u_" + FILE_NAME + ".text", u, fmt='%f')
    np.savetxt("output_files/" + "f_" + FILE_NAME + ".text", f, fmt='%f')
    np.savetxt("output_files/" + "np_nodes_" + FILE_NAME + ".text", np_nodes, fmt='%f')
    np.savetxt("output_files/" + "np_eles_" + FILE_NAME + ".text", np_eles, fmt='%f')
    np.savetxt("output_files/" + "stress_" + FILE_NAME + ".text", stress, fmt='%f')
    np.savetxt("output_files/" + "strain_" + FILE_NAME + ".text", strain, fmt='%f')
    np.savetxt("output_files/" + "disp_" + FILE_NAME + ".text", displa, fmt='%f')
    np.savetxt("output_files/" + "coords_" + FILE_NAME + ".text", coords, fmt='%f')
    np.savetxt("output_files/" + "undefc_" + FILE_NAME + ".text", undefc, fmt='%f')

if __name__ == '__main__':
    mp.freeze_support()
    main()