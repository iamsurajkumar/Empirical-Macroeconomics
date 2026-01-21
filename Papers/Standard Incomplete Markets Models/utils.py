import time

import numpy as np
from numba import guvectorize, njit

""" a. Methods to set up income and asset grids. """


def exponential_grid(x_min, x_max, num_x, order=2, pivot=1):
    """Create exponential grid of a given order."""

    # Recursively compute exp(x + log κ) - κ up to desired order.
    def _transform(x, i):
        f = np.exp(x + np.log(pivot)) - pivot
        return x_min + f if i < 2 else _transform(f, i - 1)

    # Inverse transform, used to figure out boundary.
    def _inverse_transform(x, i):
        f = np.log(x + pivot) - np.log(pivot)
        return f if i < 2 else _inverse_transform(f, i - 1)

    # Uniform grid with maximum set to implement desired x_max.
    u_max = _inverse_transform(x_max - x_min, order)
    u_grid = np.linspace(0, u_max, num_x)

    return _transform(u_grid, order)


def rouwenhorst(rho, sigma, N=7):
    """Discretize x[t] = ρx[t-1] + ϵ[t] with Rouwenhorst method.

    Parameters
    ----------
    rho   : scalar, persistence
    sigma : scalar, unconditional sd of x[t]
    N     : int, number of states in discretized Markov process

    Returns
    -------
    y  : array (N), states proportional to exp(x) s.t. E[y] = 1
    pi : array (N), stationary distribution of discretized process
    Pi : array (N*N), Markov matrix for discretized process
    """
    # Parametrize Rouwenhorst for n=2.
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # Implement recursion to build from n=3 to n=N.
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * Pi
        P2[:-1, 1:] = (1 - p) * Pi
        P3[1:, :-1] = (1 - p) * Pi
        P4[1:, 1:] = p * Pi
        Pi = P1 + P2 + P3 + P4
        Pi[1:-1] /= 2

    # Invariant distribution and scaling.
    pi = stationary(Pi)
    x = np.linspace(-1, 1, N)
    x *= sigma / np.sqrt(variance(x, pi))
    y = np.exp(x) / np.sum(pi * np.exp(x))

    return y, pi, Pi


""" b. Method for efficient linear interpolation. """


@guvectorize(
    ['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)'
)  # Use numba to vectorize and compile the code.
def interpolate(x, xq, y, yq):
    """Interpolate x -> y linearly at the query points xq.

    Both x and xq must be increasing. Interpolation is done on the _last_ dimension.
    Extrapolates linearly when xq falls outside the domain of x.

    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points

    Returns
    -------
    yq : array (nq), interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]


""" c. Young's method. """


@njit
def youngs_method(grid, x):
    """Implement Young's method via linear interpolation.

    Returns i and pi so that x = pi * grid[i] + (1-pi) * grid[i+1]

    Parameters
    ----------
    grid : array (n), ascending grid points
    x    : array (k, nx), query points (in any order)

    Returns
    -------
    i    : array (k, nx), indices of lower bracketing gridpoints
    pi   : array (k, nx), weights on lower bracketing gridpoints
    """
    x_shape = x.shape
    x = x.ravel()

    n = len(grid)
    nx = len(x)
    i = np.empty(nx, dtype=np.uint32)
    pi = np.empty(nx)

    for iq in range(nx):
        if x[iq] < grid[0]:
            ilow = 0
        elif x[iq] > grid[-2]:
            ilow = n - 2
        else:
            # use binary search to find bracketing gridpoints
            # should end with ilow and ihigh exactly 1 apart, bracketing x
            ihigh = n - 1
            ilow = 0
            while ihigh - ilow > 1:
                imid = (ihigh + ilow) // 2
                if x[iq] > grid[imid]:
                    ilow = imid
                else:
                    ihigh = imid

        i[iq] = ilow
        pi[iq] = (grid[ilow + 1] - x[iq]) / (grid[ilow + 1] - grid[ilow])

    return i.reshape(x_shape), pi.reshape(x_shape)


""" d. Auxiliary functions. """


@njit
def within_tolerance(x1, x2, tol):
    """Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2."""
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True


def stationary(Pi, tol=1e-11, maxit=10_000):
    """Find invariant distribution of a Markov chain by iteration."""
    pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    for _ in range(maxit):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    raise StopIteration(f'No convergence after {maxit} forward iterations!')


def variance(x, pi):
    """Variance of discretized random variable with support x and pmf pi."""
    return np.sum(pi * (x - np.sum(pi * x)) ** 2)


def tic(return_t=False):
    """Simple wrapper to implement matlab's tic toc methods."""
    tic.t_ = time.perf_counter()
    if return_t:
        return tic.t_


def toc(start_time=None, return_dt=False):
    """Toc method to be used in conjunction with tic."""
    if not start_time:
        start_time = tic.t_
    diff = time.perf_counter() - start_time
    if return_dt:
        return diff
    else:
        print(f'Time elapsed: {diff:.2f}s')


def shift(arr, num, fill_value=0):
    """Shift array, useful for computing leads and lags of time series variables."""
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result
