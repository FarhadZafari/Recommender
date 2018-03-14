import numpy as np
import operator
import sys

from scipy.sparse import csr_matrix

mtx = csr_matrix([[0,0,0],[0,0,0]], dtype=np.int8)
mtx = mtx.todense()

print(mtx)

t = set([1,2,3,4,4,4,4])
print(t)

np.array([1,2,3,4])

x = {2: 2, 4: 4, 3: 3, 1: 1, 0: 0}
sorted_x = sorted(x.items(), key=operator.itemgetter(1))
print(sorted_x[1:5])

y1 = {1,2,3,4,5}
y2 = {4,5,6,7,8}

y3 = y1.intersection(y2)

print(y3)

print("This is the name of the script: " + str(sys.argv[1]))
