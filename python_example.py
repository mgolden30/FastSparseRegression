import numpy as np
import python.SPRINT as SPRINT

#generate a matrix
A = np.random.random( (50,50) )

#Make an initial guess at a sparse null vector
c0 = np.zeros((50))
c0[0] = 1 #turn on first element


#add terms sequentially
n_max = 20
[coeffs, residual] = SPRINT.SPRINT_plus( A, c0, n_max)