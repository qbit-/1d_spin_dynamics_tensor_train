from __future__ import print_function, absolute_import, division
from functools import partial
import tt
import numpy as np
import sys
if '../' not in sys.path:
    sys.path.append('../')
import src as dyn


L = 4200
K = 10000
tau = 1e-1
d = 8
ranks = [5, 10, 20, 40, 256]

for r in ranks:
    H = dyn.gen_heisenberg_hamiltonian(d, Jx=1/np.sqrt(2), Jz=0)
    Mx = dyn.gen_magnetization(d)
    MxC = dyn.gen_ksite_magnetization(d)

    filenm = "output/xx/" + f"mx_mxc_d{d}_r{r}_t{tau}_L{L}.npz"
    print(f"Running XX r={r}")
    dyn.collect_ev_parallel_mpi(
        H, [Mx, MxC], dyn.gen_implicit_gaussian_guess, r, K, tau, L, filenm
    )


for r in ranks:
    H = dyn.gen_heisenberg_hamiltonian(d, Jx=0, Jz=1)
    Mx = dyn.gen_magnetization(d)
    MxC = dyn.gen_ksite_magnetization(d)

    filenm = "output/z/" + f"mx_mxc_d{d}_r{r}_t{tau}_L{L}.npz"
    print(f"Running Z r={r}")
    dyn.collect_ev_parallel_mpi(
        H, [Mx, MxC], dyn.gen_implicit_gaussian_guess, r, K, tau, L, filenm
    )

