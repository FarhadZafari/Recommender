import numpy as np
from scipy.sparse import csr_matrix

mtx = csr_matrix([[0,0,0],[0,0,0]], dtype=np.int8)
mtx = mtx.todense()

print(mtx)

t = set([1,2,3,4,4,4,4])
print(t)