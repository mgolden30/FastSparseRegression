# FastSparseRegression
This repository implements two variants of Scalable Pruning for Rapid Identification 
of Null vecTors (SPRINT). This leverages bisection to find sequential optimal rank-1 modifications
to null vectors.

# MATLAB vs Python
The code was developed in MATLAB. The python implementation is generated with a translation tool for
ease of access. 

# Convergence
SPRINT is a fast way of estimating an entire curve of sparse approximate null vectors. 
Empirically, it does a good job of finding the correct models. However, there is no convergence
gaurantee. These algorithms do a single forward or backward sweep over model complexity


Questions and comments can be directed to matthew.golden@gatech.edu
