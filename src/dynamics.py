"""
This module calculates dynamics of 1D spin chains and
saves the expectation values of the magnetization
"""
from tqdm import tqdm
import numpy as np
import tt
import os
import concurrent.futures
import functools
from tt.ksl import ksl
from collections.abc import Iterable


def collect_ev_sequential(
        H, operators, guess_generator, rank, n_samples, tau, n_steps,
        filename=None, append_file=True, dump_every=0,
        callbacks=[]):
    """
    Generate the expectation value of a provided operator
    in the dynamical process generated by the hamiltonian H.
    The dynamics starts from the initial vector, which is
    generated by the guess_generator

    Parameters:
    -----------
    H: tt.matrix
       hamiltonain matrix in the TT format
    operators: iterable of tt.matrix or tt.matrix
       matrices of the operators in the TT format
    guess_generator: function
       initial vector generator
    rank: int
       TT rank of the initial vector
    n_samples: int
       number of sample trajectories
    tau: float
       time step
    n_steps:
       number of steps of the dynamics
    filename: str, default None
       filename to output results. The file is appended if exists
    append_file: bool, default True
       if we append to the existing file instead of replacing it
    dump_every: int, default 0
       dump current results every n parallel rounds. Default is 0.
    callbacks: list, default []
       list of extra callbacks. The callback has to have a signature
       (tt.vector) -> Scalar. The callback will receive the wavefunction,
       and the result will be collected. The results of the
       callbacks are stored in the matrix along with mean values of
       the operators.

    Returns:
    --------
    (time, evs) : (np.array, np.array)
             time array and array of expectation values
    """
    # ensure that operators is iterable
    if not isinstance(operators, Iterable):
        operators = [operators]

    evs_all_l = []
    for s in tqdm(range(n_samples),
                  desc="guess={}, n_steps={}".format(
                      guess_generator.__name__, n_steps)):
        # np.random.seed(s)
        psi = guess_generator(H, rank)
        time_l = []
        evs = []
        t = 0
        psi = ksl(-1j*H, psi, 1e-10)
        for i in range(n_steps):
            ev = []
            for operator in operators:
                ev.append(tt.dot(tt.matvec(operator, psi), psi))
            for func in callbacks:
                ev.append(func(psi))

            time_l.append(t)
            evs.append(ev)

            # update
            psi = ksl(-1j*H, psi, tau)
            t += tau

        evs_all_l.append(evs)
        if (dump_every > 0) and (s // dump_every == 0):
            # time to dump results
            evs_all = np.array(evs_all_l).real
            time = np.array(time_l)
            if (s == dump_every) and (
                    not os.path.isfile(filename)
                    or not append_file):
                # rewrite old file with the first batch
                np.savez(filename, t=time,
                         evs=evs_all)
            else:
                time_old = np.load(filename)['t']
                evs_old = np.load(filename)['evs']
                assert(np.allclose(time_old, time))
                evs_updated = np.vstack((evs_old, evs_all))
                np.savez(filename, t=time, evs=evs_updated)

    evs_all = np.array(evs_all_l).real
    time = np.array(time_l)

    if filename is not None:
        if not os.path.isfile(filename) or not append_file:
            np.savez(filename, t=time, evs=evs_all)
        else:
            time_old = np.load(filename)['t']
            evs_old = np.load(filename)['evs']
            assert(np.allclose(time_old, time))
            evs_updated = np.vstack((evs_old, evs_all))
            np.savez(filename, t=time, evs=evs_updated)
    return time, evs_all


def _available_cpu_count():
    """
    Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program
    Returns:
    --------
    ncpu: int
    """
    import os
    import re
    import subprocess

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')


def parallel_worker(task_id, H, operators, guess_generator, rank,
                    tau, n_steps, callbacks=[]):
    """
    Parallel worker for the calculation of the expectation value
    of an operator

    Parameters:
    -----------
    H: tt.matrix
       hamiltonain matrix in the TT format
    operators: iterable of tt.matrix
       matrix of the operator in the TT format
    guess_generator: function
       initial vector generator
    rank: int
       TT rank of the initial vector
    tau: float
       time step
    n_steps:
       number of steps of the dynamics
    callbacks: list, default []
       list of extra callbacks. The callback has to have a signature
       (tt.vector) -> Scalar. The callback will receive the wavefunction,
       and the result will be collected. The results of the
       callbacks are stored in the matrix along with mean values of
       the operators.

    Returns:
    --------
    (time, evs) : (np.array, np.array)
             time array and array of expectation values

    """
    # np.random.seed(seed)
    psi = guess_generator(H, rank)
    time = []
    evs = []
    t = 0
    psi = ksl(-1j*H, psi, 1e-10)
    for i in range(n_steps):
        ev = []
        for operator in operators:
            ev.append(tt.dot(tt.matvec(operator, psi), psi))
        for func in callbacks:
            ev.append(func(psi))

        time.append(t)
        evs.append(ev)

        # update
        psi = ksl(-1j*H, psi, tau)
        t += tau

    evs = np.array(evs).real
    time = np.array(time)

    return time, evs


def collect_ev_parallel(H, operators, guess_generator, rank,
                        n_samples, tau, n_steps,
                        filename=None, append_file=True,
                        n_workers=None, dump_every=0,
                        callbacks=[]):
    """
    Generate expectation value using n_samples and n_steps of
    length tau in parallel

    Parameters:
    -----------
    H: tt.matrix
       hamiltonain matrix in the TT format
    operators: iterable of tt.matrix or tt.matrix
       matrix of the operator in the TT format
    guess_generator: function
       initial vector generator
    rank: int
       TT rank of the initial vector
    n_samples: int
       number of sample trajectories
    tau: float
       time step
    n_steps:
       number of steps of the dynamics
    filename: str, default None
       filename to output results. The file is appended if exists
    append_file: bool, default True
       if we append to the existing file instead of replacing it
    n_workers: int, default None
       number of parallel workers to use
    dump_every: int, default 0
       dump current results every n parallel rounds. Default is 0.
    callbacks: list, default []
       list of extra callbacks. The callback has to have a signature
       (tt.vector) -> Scalar. The callback will receive the wavefunction,
       and the result will be collected. The results of the
       callbacks are stored in the matrix along with mean values of
       the operators.

    Returns:
    --------
    (time, evs) : (np.array, np.array)
             time array and array of expectation values

    """
    # ensure that operators is iterable
    if not isinstance(operators, Iterable):
        operators = [operators]

    if n_workers is None:
        n_workers = _available_cpu_count()

    evs_all_l = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers) as executor:
        n_batches = 0
        with tqdm(desc="guess={}, n_steps={}".format(
                guess_generator.__name__, n_steps),
                  total=n_samples) as pbar:
            for time, evs in executor.map(
                    functools.partial(
                        parallel_worker,
                        H=H, operators=operators,
                        guess_generator=guess_generator,
                        rank=rank,
                        tau=tau,
                        n_steps=n_steps,
                        callbacks=callbacks),
                    range(n_samples)):
                evs_all_l.append(evs)
                pbar.update()
                n_batches += 1
                if ((dump_every > 0)
                   and (n_batches // dump_every == 0)):
                    # time to dump results
                    evs_all = np.array(evs_all_l).real
                    time = np.array(time)
                    if (n_batches == dump_every) and (
                            not os.path.isfile(filename)
                            or not append_file):
                        # rewrite old file with the first batch
                        np.savez(filename, t=time,
                                 evs=evs_all)
                    else:
                        time_old = np.load(filename)['t']
                        evs_old = np.load(filename)['evs']
                        assert(np.allclose(time_old, time))
                        evs_updated = np.vstack((evs_old, evs_all))
                        np.savez(filename, t=time, evs=evs_updated)

    evs_all = np.array(evs_all_l).real
    time = np.array(time)

    if filename is not None:
        if not os.path.isfile(filename) or not append_file:
            np.savez(filename, t=time, evs=evs_all)
        else:
            time_old = np.load(filename)['t']
            evs_old = np.load(filename)['evs']
            assert(np.allclose(time_old, time))
            evs_updated = np.vstack((evs_old, evs_all))
            np.savez(filename, t=time, evs=evs_updated)
    return time, evs_all


def _split_container(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially
    an advantage depending on the use case.
    """
    return [container[_i::count] for _i in range(count)]


def collect_ev_parallel_mpi(H, operators, guess_generator, rank,
                            n_samples, tau, n_steps,
                            filename=None, append_file=True,
                            dump_every=0, callbacks=[]):
    """
    Generate expectation value using n_samples and n_steps of length tau
    in parallel

    Parameters:
    -----------
    H: tt.matrix
       hamiltonain matrix in the TT format
    operators: iterable of tt.matrix or tt.matrix
       matrix of the operator in the TT format
    guess_generator: function
       initial vector generator
    rank: int
       TT rank of the initial vector
    n_samples: int
       number of sample trajectories
    tau: float
       time step
    n_steps:
       number of steps of the dynamics
    filename: str, default None
       filename to output results. The file is appended if exists
    append_file: bool, default True
       if we append to the existing file instead of replacing it
    dump_every: int, default 0
       dump current results every n parallel rounds. Default is 0.
    callbacks: list, default []
       list of extra callbacks. The callback has to have a signature
       (tt.vector) -> Scalar. The callback will receive the wavefunction,
       and the result will be collected. The results of the
       callbacks are stored in the matrix along with mean values of
       the operators.

    Returns:
    --------
    (time, evs) : (np.array, np.array)
             time array and array of expectation values

    """
    # ensure that operators is iterable
    if not isinstance(operators, Iterable):
        operators = [operators]

    from mpi4py import MPI
    COMM = MPI.COMM_WORLD

    if COMM.rank == 0:
        jobs = list(range(n_samples))
        # Split into however many cores are available.
        jobs = _split_container(jobs, COMM.size)
    else:
        jobs = None

    # Scatter jobs across cores.
    jobs = COMM.scatter(jobs, root=0)

    if COMM.rank == 0:
        with tqdm(desc="guess={}, n_steps={}".format(
                      guess_generator.__name__, n_steps),
                  total=len(jobs)) as pbar:
            evs_all_l = []
            for s, job in enumerate(jobs):
                # Do something meaningful here...
                time, evs = parallel_worker(
                    job, H, operators, guess_generator, rank,
                    tau, n_steps, callbacks)
                evs_all_l.append(evs)
                pbar.update()
                if ((dump_every > 0) and (
                        s // dump_every == 0) and (s != 0)):
                    # time to dump results
                    # Gather results on rank 0.
                    evs_all = MPI.COMM_WORLD.gather(evs_all_l, root=0)
                    # Flatten list of lists.
                    evs_all = [_i for temp in evs_all for _i in temp]

                    evs_all = np.array(evs_all).real
                    time = np.array(time)
                    if (s == dump_every) and (
                            not os.path.isfile(filename)
                            or not append_file):
                        # rewrite old file with the first batch
                        np.savez(filename, t=time,
                                 evs=evs_all)
                    else:
                        time_old = np.load(filename)['t']
                        evs_old = np.load(filename)['evs']
                        assert(np.allclose(time_old, time))
                        evs_updated = np.vstack((evs_old, evs_all))
                        np.savez(filename, t=time, evs=evs_updated)

    else:
        evs_all_l = []
        for s, job in enumerate(jobs):
            # Do something meaningful here...
            time, evs = parallel_worker(
                job, H, operators, guess_generator, rank,
                tau, n_steps, callbacks)
            evs_all_l.append(evs)
            if (dump_every > 0) and (s // dump_every == 0):
                # time to dump results
                # Gather results on rank 0.
                evs_all = MPI.COMM_WORLD.gather(evs_all_l, root=0)

    # Gather results on rank 0.
    evs_all = MPI.COMM_WORLD.gather(evs_all_l, root=0)

    if COMM.rank == 0:
        # Flatten list of lists.
        evs_all = [_i for temp in evs_all for _i in temp]

        evs_all = np.array(evs_all).real
        if filename is not None:
            if not os.path.isfile(filename) or not append_file:
                np.savez(filename, t=time, evs=evs_all)
            else:
                time_old = np.load(filename)['t']
                evs_old = np.load(filename)['evs']
                assert(np.allclose(time_old, time))
                evs_updated = np.vstack((evs_old, evs_all))
                np.savez(filename, t=time, evs=evs_updated)

    return time, evs_all
