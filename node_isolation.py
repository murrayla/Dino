import numpy as np

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
        [True, True, True, False],
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True]
    ]
)

TET_AREA = np.array(
    [0.5, 0.5, 0.5, 3**0.5 * (2**0.5)**2 / 4]
)

TET_NORM = np.array(
    [
        [0, 0, -1],
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, 0]
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

    return earr_2D_filt

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

    return np.array(elis_3D_filt), np.array(e_n_filt)

def main():
    # === Obtain 2D Face on Inside Cylinder Surface === #
    face_elems = simplex2D()

    # === Obtain 3D Elements with 2D Faces Located Above === #
    e_n_filt, e_n_filt = simplex3D(face_elems)

if __name__ == '__main__':
    main()
