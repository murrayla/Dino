import numpy as np
from itertools import permutations

# [Note: ζ = 1 - ξ - η]
# [
#     φ0 = ξ * (2 * ξ - 1), 
#     φ1 = η * ( 2 * η - 1),
#     φ2 = ζ * (2 * ζ - 1), 
#     φ3 = 4 * ξ * η
#     φ4 = 4 * η * ζ, 
#     φ5 = 4 * ζ * ξ
# ] @ Gauss

GP2D = np.array(
    [
        [1/3, 1/3], 
        [0.6, 0.2],
        [0.2, 0.6],
        [0.2, 0.2]
    ]
)
WE2D = np.array([-27/48, 25/48, 25/48, 25/48])
ORD2D = len(WE2D)
PHI2D = np.zeros((ORD2D, 6))
DEL2D = np.zeros((ORD2D, 3, 6))

for x in range(0, ORD2D, 1):
    # Shape Functions
    PHI2D[x, :] = np.array(
        [
            GP2D[x, 0] * (2 * GP2D[x, 0] - 1),        
            GP2D[x, 1] * (2 * GP2D[x, 1] - 1),
            (1 - GP2D[x, 0] - GP2D[x, 1]) * (2 * (1 - GP2D[x, 0] - GP2D[x, 1]) - 1),
            4 * GP2D[x, 0] * GP2D[x, 1],
            4 * GP2D[x, 1] * (1 - GP2D[x, 0] - GP2D[x, 1]),
            4 * (1 - GP2D[x, 0] - GP2D[x, 1]) * GP2D[x, 0]
        ]
    )
    # Derivatives of shape functions
    DEL2D[x, :, :] = np.array(
        [
            [4*GP2D[x, 0] - 1, 0, 4*GP2D[x, 1] + 4*GP2D[x, 0] - 3, 4*GP2D[x, 1], -4*GP2D[x, 1], 4 - 8*GP2D[x, 0] - 4*GP2D[x, 1]], 
            [0, 4*GP2D[x, 1] - 1, 4*GP2D[x, 1] + 4*GP2D[x, 0] - 3, 4*GP2D[x, 0], 4 - 8*GP2D[x, 1] - 4*GP2D[x, 0], -4*GP2D[x, 0]],
            [0, 0, 0, 0, 0, 0]
        ]
    )

TET_FACE = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
)

AREA = np.array(
    [0.5, 0.5, 0.5, 3**0.5 * (2**0.5)**2 / 4]
)

TET_AREA = {}

for i, n_id in enumerate(TET_FACE):
    for com in permutations(n_id):
        TET_AREA[com] = AREA[i]

TET_NORM = np.array(
    [
        -np.cross([1,0,0], [0,1,0]),
        -np.cross([0,0,1], [1,0,0]),
        -np.cross([0,1,0], [0,0,1]),
        np.cross([-1,1,0], [-1,0,1])
    ]
)

def simplex2D():
    # === Obtain 2D Face on Inside Cylinder Surface === #
    nlis_2D = []
    el_2D = []
    with open("runtime_files/nashAnnulus2D_cvt2dino.nodes", 'r') as node_2D:
        nlis_2D = [line.strip().split() for line in node_2D]
    with open("runtime_files/nashAnnulus2D_cvt2dino.ele", 'r') as elem_2D:
        el_2D = [line.strip().split() for line in elem_2D]
    narr_2D = np.array(nlis_2D[1:]).astype(np.float64)
    earr_2D = np.array(el_2D[1:])[:, 3:].astype(np.int32)
    n_sur = []
    for i, (x, y, _) in enumerate(narr_2D[:, 1:]):
        vec = np.linalg.norm([x, y])
        if abs(vec - 1) < 1e-5:
            n_sur.append(i)
    n_sur = set(n_sur)
    mask = np.isin(earr_2D[:, 0], list(n_sur))
    earr_2D_filt = earr_2D[mask]

    return earr_2D_filt, earr_2D

def simplex3D(face_elems):
    nlis_3D = []
    elis_3D = []
    with open("runtime_files/nashAnnulus_cvt2dino.nodes", 'r') as node_3D:
        nlis_3D = [line.strip().split() for line in node_3D]
    with open("runtime_files/nashAnnulus_cvt2dino.ele", 'r') as elem_3D:
        elis_3D = [line.strip().split() for line in elem_3D]
    narr_3D = np.array(nlis_3D[1:]).astype(np.float64)
    earr_3D = np.array(elis_3D[1:])[:, 3:].astype(np.int32)
    elis_3D_filt = []
    e_n_filt = []
    for i, row_10 in enumerate(earr_3D):
        for j, row_6 in enumerate(face_elems):
            if (all(item in row_10 for item in row_6)):
                elis_3D_filt.append(row_10)
                e_n_filt.append([i, j])

    return np.array(e_n_filt), earr_3D, narr_3D

def main():
    # === Obtain 2D Face on Inside Cylinder Surface === #
    face_elems, earr_2D = simplex2D()

    # === Obtain 3D Elements with 2D Faces Located Above === #
    e_n_filt, earr_3D, narr_3D = simplex3D(face_elems)

    # === Determine Node Positions === #
    parr = []
    f_area = []
    f_norm = []
    for i, j in e_n_filt:
        idx = [np.where(earr_3D[i] == tf)[0][0] for tf in face_elems[j]][:3]
        parr.append(idx)
        f_area.append(TET_AREA[tuple(idx)])

if __name__ == '__main__':
    main()
