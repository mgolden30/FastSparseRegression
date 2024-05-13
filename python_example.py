import numpy as np
import python.SPRINT as SPRINT

#generate a matrix
A = np.random.random( (50,50) )

#remove terms sequentially
[coeffs1, residual1] = SPRINT.SPRINT_minus( A )
print(residual1[0:6])


#add terms sequentially
#Make an initial guess at a sparse null vector
c0 = np.zeros((50))
c0[0] = 1 #turn on first element
n_max = 20
[coeffs2, residual2] = SPRINT.SPRINT_plus( A, c0, n_max)

print(residual2[0:6])
