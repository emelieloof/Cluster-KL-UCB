import numpy as np
from joblib import Parallel, delayed


def kl(p, q):
    if p == 0 and not q == 1:
        return (1 - p) * np.log(1 / (1 - q))
    if p == 1 and not q == 0:
        return p * np.log(p / q)
    if q == 0:
        if not p == 1:
            return (1 - p) * np.log((1 - p) / (1 - q))
        else:
            return 0
    if q == 1:
        if not p == 0:
            return p * np.log(p / q)
        else:
            return 0
    else:
        return (p * np.log(p / q)) + ((1 - p) * np.log((1 - p) / (1 - q)))


def one_run(algo, data_rep, T, r):
    algo.initialize(data_rep)
    for t in range(T):
        algo.select_arm()
    return np.cumsum(algo.regret)


def run_experiment(algo, data_rep, runs, T, order):
    regret = []
    bars = []
    with Parallel(n_jobs=runs) as parallel:
        regret.append(parallel(delayed(one_run)(algo, data_rep, T, r) for r in range(runs)))
    for r in regret[0]:
        error_list = []
        x = []
        for t in range(T):
            p = t + 1
            if p % 2000 == 0:
                error_list.append(r[t - ((1 + order) * 100)])
                x.append(t - ((1 + order) * 100))
        bars.append(error_list)

    c_r = [np.mean(x) for x in zip(*regret[0])]
    error = [np.std(er) for er in zip(*bars)]

    return c_r, error, x


def get_overlap(data):
    limits = []
    for i in range(len(data)):
        mini = min(data[i])
        maxi = max(data[i])
        limits.append([maxi, mini])
    overlap = 0
    max_value = max([limits[i][0] for i in range(len(data))])
    max_index = [limits[i][0] for i in range(len(data))].index(max_value)
    for i in range(len(data)):
        if limits[i][0] < max_value:
            new_overlap = limits[i][0] - limits[max_index][1]
            if new_overlap > overlap:
                overlap = new_overlap
    if not np.any(np.array([limits[j][0] for j in range(len(data))]) < limits[max_index][0]):
        return 1
    if overlap < 0:
        return 0
    return overlap

