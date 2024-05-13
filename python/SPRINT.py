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

def SPRINT_minus(G):
    """
    PURPOSE:
    Find approximate minima of L = |G*c|_2 / |c|_2 in nested sparse subspaces, such that L
    increases minimally at each stage.

    This variant of SPRINT starts with the full library and removes terms one at a time.
    See SPRINT_plus to add terms one at a time.


    INPUT:
    G - a matrix to look for sparse null vectors of



    OUTPUT:
    cs - columns of this matrix are the increasingly sparse approximate null
         vectors.
    residuals - vecnorm( G*cs )/sqrt(m);
    """

    m, n = G.shape

    cs = np.zeros((n, n))
    I = np.ones(n, dtype=bool)  # logical vector indicating sparsity
    residuals = np.zeros(n)

    if m < n:
        print("error: matrix is underdetermined.")
        return cs, residuals

    # first rotate A so it is square and upper triangular
    _, G = np.linalg.qr(G)
    G = G[:n, :n]

    # keep a copy
    A0 = G.copy()

    while n > 0:
        U, S, V = np.linalg.svd(G, full_matrices=False)
        #unlike MATLAB, S is already an array! No need to process it.

        cs[I, n-1] = V[:, -1]   # save out the smallest singular vector
        residuals[n-1] = S[-1]

        if n == 1:
            break

        candidates = np.zeros(n)
        for i in range(n):
            a = G[:, i]
            alpha = 1 / np.linalg.norm(a)
            w = alpha * U.T @ a
            ws = [w[-2], w[-1]]

            s = S
            
            bounds = [s[-1], s[-2]]

            s1, s2, s = s[-1], s[-2], s[:-2]
            w1, w2, w = w[-1], w[-2], w[:-2]

            first_term = lambda sigma: (s1**2 - sigma**2) * (s2**2 - sigma**2) * alpha**2 / (s1**2 - s2**2)
            second_term = lambda sigma: -w1**2 * (s2**2 - sigma**2) / (s1**2 - s2**2)
            third_term = lambda sigma: -w2**2 * (s1**2 - sigma**2) / (s1**2 - s2**2)
            fourth_term = lambda sigma: -np.sum(w**2 / (s**2 - sigma**2)) * (s1**2 - sigma**2) * (s2**2 - sigma**2) / (s1**2 - s2**2)

            r = s1 / s2
            first_term = lambda sigma: (r**2 - (sigma/s2)**2) * (s2**2 - sigma**2) * alpha**2 / (r**2 - 1)
            second_term = lambda sigma: -w1**2 * (1**2 - (sigma/s2)**2) / (r**2 - 1)
            third_term = lambda sigma: -w2**2 * (r**2 - (sigma/s2)**2) / (r**2 - 1)
            fourth_term = lambda sigma: -np.sum(w**2 / ((s/s2)**2 - (sigma/s2)**2)) * (r**2 - (sigma/s2)**2) * (1 - (sigma/s2)**2) / (r**2 - 1)

            f = lambda sigma: first_term(sigma) + second_term(sigma) + third_term(sigma) + fourth_term(sigma)

            maxit = 128
            threshold = 1e-130
            g = 0

            for _ in range(maxit):
                g = np.sum(bounds) / 2  # bisection guess
                fg = f(g)
                if abs(fg) < threshold:
                    break

                if fg > 0:
                    bounds[0] = g
                else:
                    bounds[1] = g
            candidates[i] = g

        i_min = np.argmin(candidates)
        j = np.where(I)[0]
        I[j[i_min]] = False
        G = A0[:, I]
        n -= 1

    # rescale the residual
    residuals /= np.sqrt(m)
    return cs, residuals

