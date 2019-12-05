"""
This module implements different initial guesses for
the TT wavefunction
"""

import numpy as np
import tt
from tt.riemannian.riemannian import project as tt_project
from tt.riemannian.riemannian import tt_qr


def haar_random_unitary(n, m):
    """
    Generates m columns of a random unitary
    of size n distributed according to Haar measure.
    Parameters:
    -----------
    n: int
       size of the matrix
    m: int
       number of columns
    """
    assert(m <= n)
    uf = np.random.randn(n, n)
    u, r = np.linalg.qr(uf)
    lam = np.diag(np.diag(r) / (abs(np.diag(r))))
    return (u @ lam)[:, :m]


def gen_haar_cores_like(t, left_to_right=True):
    """
    Generates Haar distributed cores for a TT tensor
    of shape provided by t
    Parameters:
    ----------
    t: tt.tensor
         tensor to get shape information
    left_to_right: bool, default True
         left to right or right to left orthogonalization
    Returns:
    --------
    list
    """
    tl = t.to_list(t)
    haar_cores = []
    for core in tl:
        if left_to_right:
            matrix_shape = [np.prod(core.shape[:-1]), core.shape[-1]]
            haar_u = haar_random_unitary(*matrix_shape)
        else:
            matrix_shape = [np.prod(core.shape[1:]), core.shape[0]]
            haar_u = haar_random_unitary(*matrix_shape)
            haar_u = haar_u.T
        haar_cores.append(haar_u.reshape(core.shape))
    return haar_cores


def gen_zero_energy_guess(H, rank):
    """
    Generate psi such that <psi|H|psi> = 0
    Parameters:
    -----------
    H: tt.matrix
       hamiltonian in the TT-matrix format
    rank: int
       Rank of the guess
    """
    v = 1.0
    while v > 1e-12:
        # Create two random TT vectors and normalize them
        psi1 = tt.rand(H.n, r=rank)
        psi2 = tt.rand(H.n, r=1)
        psi1 = psi1*(1.0/psi1.norm())
        psi2 = psi2*(1.0/psi2.norm())
        # Calculate coefficients of the quadratic equation
        h22 = tt.dot(tt.matvec(H, psi2), psi2)
        h21 = tt.dot(tt.matvec(H, psi2), psi1)
        h11 = tt.dot(tt.matvec(H, psi1), psi1)
        # find vectors such that <psi|H|psi> = 0
        rs = np.roots([h22, 2*h21, h11])
        v = np.linalg.norm(np.imag(rs))

    psi = psi1 + rs[0]*psi2
    psi = psi*(1.0/psi.norm())
    return psi


def project_gaussian_to_x(X):
    """
    Generates projection of a the gaussian vector in tangent space
    to a given TT vector X

    Parameters:
    -----------
    X: tt.vector
       Vector to project to
    """
    x = X.round(eps=1e-14)  # removes excessive ranks
    d = x.d
    r = x.r
    dimensions = x.n
    Z = np.random.randn(*(dimensions))
    Z = tt.tensor(Z)
    PZ = tt_project(x, Z)
    return PZ


def project_distribution_to_x_implicit(X, sampling_function):
    """
    Generates an implicit projection of the vector in tangent
    space to a given TT vector X. The cores of a random TT vector
    are generated by the sampling_function

    Parameters:
    -----------
    X: tt.vector
       Vector to project to
    """
    l_cores = tt.vector.to_list(
        tt_qr(X, left_to_right=True)[0])
    r_cores = tt.vector.to_list(
        tt_qr(X, left_to_right=False)[0])
    Z_random_cores = [sampling_function(l_cores[ii].shape)
                      for ii in range(X.d)]

    # generate edge terms
    PZ = tt.vector.from_list([Z_random_cores[0]] + r_cores[1:])
    PZ += tt.vector.from_list(l_cores[:X.d-1] + [Z_random_cores[X.d-1]])

    # generate sum from the exact expression
    for ii in range(1, X.d-2):
        PZ += tt.vector.from_list(l_cores[:ii] +
                                  [Z_random_cores[ii]] + r_cores[ii+1:])
    return PZ


def project_gaussian_to_x_implicit(X):
    """
    Generates an implicit projection
    of the gaussian vector in
    tangent space to a given TT vector X. The cores of TT
    random vector are sampled from the normal distribution

    Parameters:
    -----------
    X: tt.vector
       Vector to project to
    """
    PZ = project_distribution_to_x_implicit(
        X, lambda tup: np.random.randn(*tup))
    return PZ


def gen_implicit_gaussian_guess(H, r):
    """
    Generates an implicit projection
    of a normal vector in
    tangent space to a random TT
    (with cores drawn from normal distribution)

    Parameters:
    -----------
    H: tt.matrix
       Matrix used to infer dimension of a guess vector
    r: int
       TT rank of the guess
    """
    dimensions = H.n
    X = tt.rand(dimensions, r=r).round(1e-14)
    psi = project_gaussian_to_x_implicit(X)
    psi = psi.round(0, rmax=r)
    psi = 1./psi.norm() * psi
    return psi


def half_normal_distribution(shape):
    """
    Generates a vector X such that
    <X|X> = N(0, 1) (elementwise).
    X has a shape provided by the argument.

    Parameters:
    -----------
    shape: tuple
       Shape of the guess
    """
    y = np.random.randn(*shape)
    sign = np.sign(y)
    return sign * np.exp(1/2 * np.log(np.abs(y)))


def gen_implicit_guess_from_distrib(
        H, r,
        sample_function=half_normal_distribution,
        normalize=True):
    """
    Generates an implicit projection of a vector in
    tangent space of a random TT (with cores drawn
    from normal distribution). The vector is generated by the
    sample_function

    Parameters:
    -----------
    H: tt.matrix
       Matrix used to infer dimension of a guess vector
    r: int
       TT rank of the guess
    """
    dimensions = H.n
    X = tt.rand(dimensions, r=r).round(1e-14)
    psi = project_distribution_to_x_implicit(X, sample_function)
    psi.round(0, rmax=r)
    if normalize:
        psi = 1./psi.norm() * psi
    return psi


def gen_rounded_gaussian_guess(H, r, eps=1e-10):
    """
    Generate full N(0,1) vector and then
    compress it to a TT with given rank and eps

    Parameters:
    -----------
    H: tt.matrix
       Matrix used to infer dimension of a guess vector
    r: int
       TT rank of the guess
    """
    psi_full = np.random.randn(*H.n)
    psi = tt.tensor(psi_full, eps, rmax=r)
    psi = psi*(1.0/psi.norm())
    return psi


def gen_projected_gaussian_guess(H, r, eps=1e-10):
    """
    Generate full N(0,1) vector and then
    project it to a random unitary TT of given rank

    Parameters:
    -----------
    H: tt.matrix
       Matrix used to infer dimension of a guess vector
    r: int
       TT rank of the guess
    """
    # generate tensor with haar distributed cores
    x = tt.rand(H.n, r=r)
    x = x.round(eps)
    x_cores = gen_haar_cores_like(x, left_to_right=True)
    x = x.from_list(x_cores)

    # project full dimensional gaussian vector to x
    dimensions = x.n
    Z = np.random.randn(*(dimensions))
    Z = tt.tensor(Z)
    PZ = tt_project(x, Z)
    PZ = PZ.round(eps, rmax=r)

    return PZ*(1.0 / PZ.norm())


def gen_haar_rmps_guess(H, r):
    """
    Generates a random MPS as described in
    "Typicality in random matrix product states" by Garnerone et al. 

    Parameters:
    -----------
    H: tt.matrix
       Matrix used to infer dimension of a guess vector
    r: int
       TT rank of the guess
    """
    # generate tensor with haar distributed cores
    x = tt.rand(H.n, r=r)
    x_cores = gen_haar_cores_like(x, left_to_right=True)
    x = x.from_list(x_cores)

    return x
