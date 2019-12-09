"""
This module generates various problem-specific quantities:
Hamiltonian, Magnetization operator etc
"""

import numpy as np
import tt


def gen_1site_1d_operator(matrix, spacer, site_num, dimension):
    """
    Generate 1D operator in TT form.
    Parameters
    ----------
    matrix: tt.matrix
         Matrix of the operator (which acts upon the site)
    spacer: tt.matrix
         Spacer used for all other sites (usually identity)
    site_num: int
         Number of site where matrix acts
    dimension: total number of sites
    """
    result = matrix
    for j in range(site_num):
        result = tt.kron(spacer, result)
    for j in range(dimension-site_num-1):
        result = tt.kron(result, spacer)
    return result


def gen_heisenberg_hamiltonian(dimension, Jx=0.5, Jz=0.5,
                               periodic=True):
    """
    Generate 1D Heisenberg hamiltonian
    Parameters:
    -----------
    dimension: int
           Numner of sites
    Jx: float
           Sx coupling strength
    Jz: float
           Sz coupling strength
    periodic: bool
           If the Hamiltonian will be made periodic
    """
    # Create operators
    sx = [[0, 1], [1, 0]]
    sx = np.array(sx, dtype=np.float)
    sz = [[1, 0], [0, -1]]
    sz = np.array(sz, dtype=np.float)
    sz = 0.5 * sz
    sp = [[0, 1], [0, 0]]
    sp = np.array(sp, dtype=np.float)
    sm = sp.T
    e = np.eye(2)
    sx = tt.matrix(sx, 1e-12)
    sz = tt.matrix(sz, 1e-12)
    sp = tt.matrix(sp, 1e-12)
    sm = tt.matrix(sm, 1e-12)
    e = tt.matrix(e, 1e-12)
    # Generate arrays of 1 site operators
    ssp = [gen_1site_1d_operator(sp, e, i, dimension)
           for i in range(dimension)]
    ssz = [gen_1site_1d_operator(sz, e, i, dimension)
           for i in range(dimension)]
    ssm = [gen_1site_1d_operator(sm, e, i, dimension)
           for i in range(dimension)]
    ssx = [gen_1site_1d_operator(sx, e, i, dimension)
           for i in range(dimension)]
    Ham = None
    for i in range(dimension-1):
        Ham = Ham + 0.5 * Jx * (ssp[i] * ssm[i+1] + ssm[i]
                                * ssp[i+1]) + Jz * (ssz[i] * ssz[i+1])
        # compress
        Ham = Ham.round(1e-8)
    # Add periodic conditions in the Hamiltonian
    if periodic:
        Ham = Ham + 0.5 * Jx * (ssp[dimension-1] * ssm[0] + ssm[dimension-1] *
                                ssp[0]) + Jz * (ssz[dimension-1] * ssz[0])
    Ham = Ham.round(1e-8)
    return Ham


def gen_ksite_magnetization(dimension, used_sites=None, scale_sites=None):
    """
    Generate the magnetization operator Mx

    Parameters:
    -----------
    dimension: int
           Number of sites
    used_sited: np.array, default None
           Logical array indicating which sites to measure
           By default a central site is used for odd
           dimension and a central pair of sites for even dimension
    scale_sites: np.array, default None
           Array for setting weight of every site. Default 0.
           Should have the same size as dimension
    """
    # Decide which sites to use
    if used_sites is None:
        used_sites = np.zeros(dimension)
        if dimension % 2 == 0:
            used_sites[dimension // 2] = 1
            used_sites[dimension // 2 - 1] = 1
        else:
            used_sites[dimension // 2] = 1

    if scale_sites is not None:
        assert(len(scale_sites) == dimension)
    else:
        scale_sites = np.ones(dimension)

    # Create operators
    sx = [[0, 1], [1, 0]]
    sx = np.array(sx, dtype=np.float)
    e = np.eye(2)
    sx = tt.matrix(sx, 1e-12)
    e = tt.matrix(e, 1e-12)
    # Generate arrays of 1 site operators
    ssx = [gen_1site_1d_operator(sx, e, i, dimension)
           for i in range(dimension)]

    Mx = None
    for i in range(dimension):
        if bool(used_sites[i]) is True:
            Mx = Mx + 0.5 * ssx[i] * scale_sites[i]
            Mx = Mx.round(1e-8)
    return Mx


def gen_magnetization(dimension):
    """
    Generate the magnetization operator Mx

    Parameters:
    -----------
    dimension: int
           Number of sites
    """
    return gen_ksite_magnetization(dimension, np.ones(dimension))


def gen_scaled_heisenberg_hamiltonian(dimension, cluster_size,
                                      Jx=0.5, Jz=0.5,
                                      periodic=True):
    """
    Generate 1D Heisenberg hamiltonian
    Parameters:
    -----------
    dimension: int
           Number of sites
    cluster_size: int
           Size of the cluster which is simulated exactly
           The cluster will be placed in the center of the chain
    Jx: float
           Sx coupling strength
    Jz: float
           Sz coupling strength
    periodic: bool
           If the Hamiltonian will be made periodic
    """
    # import pdb
    # pdb.set_trace()
    assert(dimension >= cluster_size)

    # Create operators
    sx = [[0, 1], [1, 0]]
    sx = np.array(sx, dtype=np.float)
    sz = [[1, 0], [0, -1]]
    sz = np.array(sz, dtype=np.float)
    sz = 0.5 * sz
    sp = [[0, 1], [0, 0]]
    sp = np.array(sp, dtype=np.float)
    sm = sp.T
    e = np.eye(2)
    sx = tt.matrix(sx, 1e-12)
    sz = tt.matrix(sz, 1e-12)
    sp = tt.matrix(sp, 1e-12)
    sm = tt.matrix(sm, 1e-12)
    e = tt.matrix(e, 1e-12)
    # Generate arrays of 1 site operators
    ssp = [gen_1site_1d_operator(sp, e, i, dimension)
           for i in range(dimension)]
    ssz = [gen_1site_1d_operator(sz, e, i, dimension)
           for i in range(dimension)]
    ssm = [gen_1site_1d_operator(sm, e, i, dimension)
           for i in range(dimension)]
    ssx = [gen_1site_1d_operator(sx, e, i, dimension)
           for i in range(dimension)]
    Ham = None

    space_size = 2**(cluster_size // 2)
    half_no_cluster = (dimension - cluster_size) // 2
    cluster_start = half_no_cluster - 1 if half_no_cluster > 0 else 0

    # classical subsystem
    for i in range(cluster_start):
        Ham = Ham + 0.5 * Jx * (ssp[i] * ssm[i+1] + ssm[i]
                                * ssp[i+1]) + Jz * (ssz[i] * ssz[i+1])

    for i in range(dimension - half_no_cluster, dimension - 1):
        Ham = Ham + 0.5 * Jx * (ssp[i] * ssm[i+1] + ssm[i]
                                * ssp[i+1]) + Jz * (ssz[i] * ssz[i+1])
    # quantum cluster
    center_of_cluster = cluster_start + cluster_size // 2 - 1
    cluster_end = cluster_start + cluster_size - 1
    if half_no_cluster > 0:
        cluster_end += 2
        center_of_cluster += 1

    jj = 0
    # if the cluster coincides with the system correct
    # the indices and the exponent counter
    if half_no_cluster == 0:
        jj += 1

    for i in range(cluster_start, cluster_end):
        scaling_factor = 1 / np.sqrt(space_size / 2**jj + 1)
        Ham = Ham + 0.5 * Jx * scaling_factor * (
            ssp[i] * ssm[i+1] + ssm[i]
            * ssp[i+1]) + Jz * scaling_factor * (ssz[i] * ssz[i+1])
        if i < center_of_cluster:
            jj += 1
        else:
            jj -= 1
        # compress
        Ham = Ham.round(1e-8)

    # Add periodic conditions in the Hamiltonian
    if periodic:
        # the cluster coincides with the system size. Maximal
        # scaling factor across the edge
        if half_no_cluster == 0:
            scaling_factor = 1 / np.sqrt(space_size + 1)
            Ham = Ham + 0.5 * Jx * scaling_factor * (
                ssp[dimension-1] * ssm[0] + ssm[dimension-1] *
                ssp[0]) + Jz * (ssz[dimension-1] * ssz[0])
        else:
            Ham = Ham + 0.5 * Jx * (
                ssp[dimension-1] * ssm[0] + ssm[dimension-1] *
                ssp[0]) + Jz * (ssz[dimension-1] * ssz[0])
    Ham = Ham.round(1e-8)
    return Ham


def gen_scaling_factors(dimension, cluster_size):
    """
    Generate scaling factors for cluster
    Parameters:
    -----------
    dimension: int
           Number of sites
    cluster_size: int
           Size of the cluster which is simulated exactly
           The cluster will be placed in the center of the chain

    """
    # import pdb
    # pdb.set_trace()
    assert(dimension >= cluster_size)

    space_size = 2**(cluster_size // 2)
    half_no_cluster = (dimension - cluster_size) // 2
    cluster_start = half_no_cluster - 1 if half_no_cluster > 0 else 0

    scaling_factors = np.zeros(dimension)
    # classical subsystem
    for i in range(cluster_start):
        scaling_factors[i] = 1

    for i in range(dimension - half_no_cluster, dimension - 1):
        scaling_factors[1] = 1

    # quantum cluster
    center_of_cluster = cluster_start + cluster_size // 2 - 1
    cluster_end = cluster_start + cluster_size - 1
    if half_no_cluster > 0:
        cluster_end += 2
        center_of_cluster += 1

    jj = 0
    # if the cluster coincides with the system correct
    # the indices and the exponent counter
    if half_no_cluster == 0:
        jj += 1

    for i in range(cluster_start, cluster_end):
        scaling_factor = 1 / np.sqrt(space_size / 2**jj + 1)
        scaling_factors[i] = scaling_factor

    return scaling_factors


def gen_scaled_magnetization(dimension, cluster_size):
    """
    Generate scaled magnetization operator for cluster
    Parameters:
    -----------
    dimension: int
           Number of sites
    cluster_size: int
           Size of the cluster which is simulated exactly
           The cluster will be placed in the center of the chain
    """
    scaling_factors = gen_scaling_factors(dimension, cluster_size)
    return gen_ksite_magnetization(dimension, np.ones(dimension),
                                   scaling_factors)
