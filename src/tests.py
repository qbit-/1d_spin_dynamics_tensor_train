#!/usr/bin/env python

# ## this file simply tests functions in the src directory

import sys
if '../' not in sys.path:
    sys.path.append('../')
import src as dyn
import numpy as np
import tt


def gen_1d(mat, e, i, d):
    """Generate 1D operator in TT form"""
    w = mat
    for j in range(i):
        w = tt.kron(e,w)
    for j in range(d-i-1):
        w = tt.kron(w,e)
    return w


def gen_heisen(d, Jx=0.5, Jz=0.5):
    """Generate Heisenberg Hamiltonian and Mx operator"""
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
    # Generate ssx, ssz
    ssp = [gen_1d(sp, e, i, d) for i in range(d)]
    ssz = [gen_1d(sz, e, i, d) for i in range(d)]
    ssm = [gen_1d(sm, e, i, d) for i in range(d)]
    ssx = [gen_1d(sx, e, i, d) for i in range(d)]
    A = None
    Mx = None
    for i in range(d-1):
        A = A + 0.5 * Jx * (ssp[i] * ssm[i+1] + ssm[i] * ssp[i+1]) + Jz * (ssz[i] * ssz[i+1])
        A = A.round(1e-8)
        Mx = Mx + 0.5 * ssx[i]
        Mx = Mx.round(1e-8)
    A = A + 0.5 * Jx * (ssp[d-1] * ssm[0] + ssm[d-1] * ssp[0]) +  Jz * (ssz[d-1] * ssz[0])
    A = A.round(1e-8)
    Mx = Mx + 0.5 * ssx[d-1]
    Mx = Mx.round(1e-8)
    return A, Mx


def test_operators():
    d = 3

    # operators
    H1 = dyn.gen_heisenberg_hamiltonian(d)
    Mx1 = dyn.gen_magnetization(d)

    H2, Mx2 = gen_heisen(d)

    assert(np.allclose(H1.full(), H2.full()))
    assert(np.allclose(Mx1.full(), Mx2.full()))


def test_guess():
    # Set up the parameters of the script
    d = 4
    r = 5

    H = dyn.gen_heisenberg_hamiltonian(d)
    # guess
    np.random.seed(10)
    psi1 = dyn.gen_rounded_gaussian_guess(H, r)
    np.random.seed(10)
    psi2 = dyn.gen_implicit_gaussian_guess(H, r)
    assert(np.allclose(tt.dot(psi1, psi2), -0.18779753088022963))


def test_dynamics():
    # Set up the parameters of the script
    d = 4
    K = 4
    r = 5
    tau = 1e-1
    L = 42

    H = dyn.gen_heisenberg_hamiltonian(d)
    Mx = dyn.gen_magnetization(d)

    np.random.seed(42)
    t1, xmx1 = dyn.collect_ev_sequential(
        H, Mx, dyn.gen_implicit_gaussian_guess, r, K, tau, L)

    np.random.seed(42)
    t2, xmx2 = dyn.collect_ev_parallel(
        H, Mx, dyn.gen_implicit_gaussian_guess, r, K, tau, L, n_workers=2)

    np.random.seed(42)
    t3, xmx3 = dyn.collect_ev_parallel_mpi(
        H, Mx, dyn.gen_implicit_gaussian_guess, r, K, tau, L)

    assert(np.allclose(xmx1[0, :], xmx2[0, :]))
    assert(np.allclose(xmx2[0, :], xmx3[0, :]))

if __name__ == '__main__':
    test_operators()
    test_guess()
    test_dynamics()
