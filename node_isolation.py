import numpy as np

GP = np.array(
    [
        [1/3, 1/3], 
        [0.6, 0.2],
        [0.2, 0.6],
        [0.2, 0.2]
    ]
)
WE = np.array([-27/48, 25/48, 25/48, 25/48])
ORDER = len(WE)

#   η
#   1  2                   
#   |  |`\                 
#   |  |  `\               
#   |  5    `4             
#   |  |      `\            
#   |  |        `\          
#   0  0-----3-----1
#      0-----------1 ξ


# [Note: ζ = 1 - ξ - η]
# [
#     φ0 = ξ * (2 * ξ - 1), 
#     φ1 = η * ( 2 * η - 1),
#     φ2 = ζ * (2 * ζ - 1), 
#     φ3 = 4 * ξ * η
#     φ4 = 4 * η * ζ, 
#     φ5 = 4 * ζ * ξ
# ] @ Gauss

PHI = np.zeros((ORDER, 6))

# [
#     δφ1/δξ δφ2/δξ ... δφ6/δξ
#     δφ1/δη δφ2/δη ... δφ6/δη
# ] @ Gauss

DEL_PHI = np.zeros((ORDER, 2, 6))

for x in range(0, ORDER, 1):
    # Shape Functions
    PHI[x, :] = np.array(
        [
            GP[x, 0] * (2 * GP[x, 0] - 1),        
            GP[x, 1] * (2 * GP[x, 1] - 1),
            (1 - GP[x, 0] - GP[x, 1]) * (2 * (1 - GP[x, 0] - GP[x, 1]) - 1),
            4 * GP[x, 0] * GP[x, 1],
            4 * GP[x, 1] * (1 - GP[x, 0] - GP[x, 1]),
            4 * (1 - GP[x, 0] - GP[x, 1]) * GP[x, 0]
        ]
    )
    # Derivatives of shape functions
    DEL_PHI[x, :, :] = np.array(
        [
            [4*GP[x, 0] - 1, 0, 4*GP[x, 1] + 4*GP[x, 0] - 3, 4*GP[x, 1], -4*GP[x, 1], 4 - 8*GP[x, 0] - 4*GP[x, 1]], 
            [0, 4*GP[x, 1] - 1, 4*GP[x, 1] + 4*GP[x, 0] - 3, 4*GP[x, 0], 4 - 8*GP[x, 1] - 4*GP[x, 0], -4*GP[x, 0]]
        ]
    )

nodes = open("runtime_files/nashAnnulus2D_cvt2dino.nodes", 'r')
elems = open("runtime_files/nashAnnulus2D_cvt2dino.ele", 'r')
n_list = list()
e_list = list()

for line in nodes:
    n_list.append(line.strip().replace('\t', ' ').split(' '))
for line in elems:
    e_list.append(line.strip().replace('\t', ' ').split(' '))

np_n = np.array(n_list[1:]).astype(np.float64)
np_e = np.array(e_list[1:])
np_e = np_e[:, 3:].astype(np.int32)
n_ele = len(np_e[:, 0])
n_n = int(len(np_n[:, 0]))
dim = 3

n_sur = list()

for i, (x, y, z) in enumerate(np_n[:, 1:]):
    vecNorm = np.linalg.norm([x, y])
    if abs(vecNorm - 1) < 1e-3:
        n_sur.append(i)

n_sur = set(n_sur)
mask = np.isin(np_e[:, 0], list(n_sur))
filt_np_e = np_e[mask]

