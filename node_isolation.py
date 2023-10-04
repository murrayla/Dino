import numpy as np

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
print(filt_np_e)
