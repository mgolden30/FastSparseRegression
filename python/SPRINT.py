import numpy as np

def SPRINT_plus(G, c0, n_max):
    """
    PURPOSE:
    The minimum of L = ||G*c||_2 / ||c||_2 in nested sparse subspaces,
    such that L decreases maximally at each stage.

    This variant of SPRINT starts with a sparse model c0 and adds terms one at a time.

    INPUT:
    G - a matrix to look for sparse null vectors of
    c0 - initial sparse model
    n_max - maximum number of iterations

    OUTPUT:
    cs - columns of this matrix are the increasingly sparse approximate null vectors.
    residuals - vecnorm( G*cs );
    """
    m, n = G.shape
    if m < n:
        print("error: matrix is underdetermined.")
        return None, None

    # First rotate G so it is square and upper triangular
    Q, R = np.linalg.qr(G)
    G = R[:n, :n]

    cs = np.zeros((n, n))
    I = (c0 != 0)  # logical vector indicating sparsity
    residuals = np.zeros(n)

    while np.sum(I) < n_max:
        ns = np.sum(I)

        #Since I wrote this with MATLAB indexing, subtract 1 from ns to use it as an index
        ns = ns - 1

        U, S, Vt = np.linalg.svd(G[:, I], full_matrices=False)
        cs[I, ns] = Vt[ns, :]  # save out the smallest singular vector
        residuals[ns] = S[ns]

        candidates = np.zeros(n)
        for i in range(n):
            if I[i] == 1:
                candidates[i] = np.inf
                continue

            a = G[:, i]
            alpha = 1 / np.linalg.norm(a)
            w = alpha * U.T @ a

            s = S  # singular values array
            bounds = [0, s[-1]]  # must be lower than current singular value

            tau = np.sqrt(1 - np.sum(w ** 2))
            f0 = lambda sigma: 1 + 1/alpha**2 * np.sum(w**2 / (s**2 - sigma**2)) - tau**2 / alpha**2 / sigma**2
            reg = lambda sigma: sigma**2 * (s[-1]**2 - sigma**2) * alpha**2
            f = lambda sigma: f0(sigma) * reg(sigma)

            maxit = 128
            threshold = 1e-130

            for _ in range(maxit):
                g = np.sum(bounds) / 2  # bisection guess
                fg = f(g)

                if abs(fg) < threshold:
                    break

                if fg < 0:
                    bounds[0] = g
                else:
                    bounds[1] = g

            candidates[i] = g

        i_min = np.argmin(candidates)
        I[i_min] = 1  # include this term

    # Rescale the residual
    residuals = residuals / np.sqrt(m)

    return cs, residuals
