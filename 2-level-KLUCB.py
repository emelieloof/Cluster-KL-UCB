import numpy as np
from joblib import Parallel, delayed
import random as rnd
import matplotlib.pyplot as plt
import json


class TwolevelKLUCB:

    def __init__(self, data):
        self.steps = 1
        counts = []
        rewards = []
        for i in range(len(data)):
            counts.append([0] * len(data[i]))
            rewards.append([0] * len(data[i]))
        self.counts = counts
        self.rewards = rewards
        self.regret = []

        cluster_belongings = [0] * np.sum([len(i) for i in self.rewards])
        index = 0
        for i in range(len(self.rewards)):
            for j in range(len(self.rewards[i])):
                cluster_belongings[index] = i
                index += 1
        self.cluster_belongings = cluster_belongings

        arm_belongings = [0] * np.sum([len(i) for i in self.rewards])
        index = 0
        for i in range(len(self.rewards)):
            for j in range(len(self.rewards[i])):
                arm_belongings[index] = j
                index += 1
        self.arm_belongings = arm_belongings

    def initialize(self, data):
        self.steps = 1
        counts = []
        rewards = []
        for i in range(len(data)):
            counts.append([0] * len(data[i]))
            rewards.append([0] * len(data[i]))
        self.counts = counts
        self.rewards = rewards
        self.regret = []

    def get_means(self):
        means = []
        for i in range(len(self.counts)):
            curr = []
            for j in range(len(self.counts[i])):
                curr.append(self.rewards[i][j] / self.counts[i][j])
            means.append(curr)
        return means

    def get_bounds(self, means, level, cluster, t):
        Q = []
        for i in range(len(means)):
            if level == 1:
                if max(means[i]) == 1:
                    Q.append(1.0)
                else:
                    q = np.linspace(max(means[i]), 1, 50)
                    low = 0
                    high = len(q) - 1
                    while high > low:
                        mid = round((low + high) / 2)
                        prod = self.get_sum(means[i], q[mid], i)
                        if prod < np.log(t):
                            low = mid + 1
                        else:
                            high = mid - 1
                    Q.append(q[low])

            else:
                if means[i] == 1:
                    Q.append(1.0)
                else:
                    q = np.linspace(means[i], 1, 50)
                    low = 0
                    high = len(q) - 1
                    while high > low:
                        mid = round((low + high) / 2)
                        prod = np.sum(self.counts[cluster][i] * (kl(means[i], q[mid])))
                        if prod < np.log(t):
                            low = mid + 1
                        else:
                            high = mid - 1
                    Q.append(q[low])
        return Q

    def get_sum(self, means, q, index):
        s = 0
        for i in range(len(means)):
            s += self.counts[index][i] * kl(means[i], q)
        return s

    def select_cluster(self, t):
        if t < np.sum([len(i) for i in self.rewards]):
            cluster = self.cluster_belongings[t]
        else:
            means = self.get_means()
            curr_opt_c = means.index(max(means))
            bounds = self.get_bounds(means, 1, None, t)
            cluster = self.get_max_index(bounds, curr_opt_c)
        return cluster

    def select_arm(self, cluster, t):
        if t < np.sum([len(i) for i in self.rewards]):
            arm = self.arm_belongings[t]
        else:
            means = self.get_means()
            curr_opt_a = means[cluster].index(max(means[cluster]))
            bounds = self.get_bounds(means[cluster], 2, cluster, t)
            arm = self.get_max_index(bounds, curr_opt_a)
        reward = bandit.draw(cluster, arm)
        regret = bandit.regret(cluster, arm)
        self.update(arm, cluster, reward, regret)

    def get_max_index(self, l, optimal):
        true_opt = np.max(l)
        true_index = l.index(true_opt)
        for i in range(len(l)):
            if l[i] == true_opt and i == optimal:
                true_index = i
        return true_index

    def reshape(self, list):
        new_list = np.zeros(len(self.arm_belongings))
        index = 0
        for i in range(len(list)):
            for j in range(len(list[i])):
                new_list[index] = list[i][j]
                index += 1
        return new_list

    def update(self, arm, cluster, reward, regret):
        self.steps += 1
        self.counts[cluster][arm] += 1
        self.rewards[cluster][arm] += reward
        self.regret.append(regret)


def kl(p, q):  # function to get the kl-divergence between p and q
    epsilon = 0.0001
    if p == 0:
        p += epsilon
    elif p == 1:
        p -= epsilon
    if q == 0:
        q += epsilon
    elif q == 1:
        q -= epsilon
    result = (p * np.log(p / q)) + ((1 - p) * np.log((1 - p) / (1 - q)))
    return result


def one_run(algo, data_rep, T, r):
    algo.initialize(data_rep)
    for t in range(T):
        cluster = algo.select_cluster(t)
        algo.select_arm(cluster, t)
        if t % 1000 == 0:
            print(t)
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


class BernoulliBandit:  # A class which creates an actual bandit out of a Bernoulli arm,
    def __init__(self, theta):
        subopt = []
        for i in range(len(theta)):
            subopt.append([x1 - x2 for (x1, x2) in zip([np.max(theta)] * len(theta[i]), theta[i])])
        self.sub_opt = subopt
        arms = []
        for i in range(len(theta)):
            arms.append(list(map(lambda m: BernoulliArm(m), theta[i])))
        self.arms = arms

    def draw(self, cluster, arm):
        return self.arms[cluster][arm].draw()

    def regret(self, cluster, arm):
        return self.sub_opt[cluster][arm]


class BernoulliArm:  # A class to create an arm and draw a reward from its Bernoulli distribution
    def __init__(self, p):
        self.p = p

    def draw(self):
        if rnd.random() > self.p:
            return 0
        else:
            return 1


T = 10000
runs = 50


def generate_data(limits_opt, dist_to_opt, num_c, num_a):
    data = []
    opt_c = [round(i, 2) for i in np.linspace(limits_opt[0], limits_opt[1], num_a)]
    data.append(opt_c)
    sub_c = [round(i - dist_to_opt, 2) for i in opt_c]
    for i in range(1, num_c):
        data.append(sub_c)
    return data


data_75 = generate_data([0.8, 0.6], 0.05, 5, 5)
data_25 = [data_75[0]]
data_SD = [data_75[0]]
for i in range(1, 5):
    data_25.append([round(j - 0.1, 2) for j in data_75[1]])
    data_SD.append([round(j - 0.2, 2) for j in data_75[1]])

data_big_opt_c = [[round(i, 2) for i in np.linspace(0.8, 0.4, 5)]]
worst_optc = [0.8]
list_c = np.linspace(0.4, 0.2, 4)
data_worst = [worst_optc]
for i in range(4):
    worst_optc.append(round(list_c[i], 2))
for i in range(4):
    data_worst.append([round(j - 0.1, 2) for j in data_75[0]])
    data_big_opt_c.append([round(j - 0.1, 2) for j in data_75[0]])

d = [data_75, data_25, data_SD, data_big_opt_c, data_worst]
result_string = {'2level KLUCB': []}

for data in d:
    bandit = BernoulliBandit(data)
    data_rep = []
    for i in range(len(data)):
        data_rep.append([0] * len(data[i]))
    two_level_kl = TwolevelKLUCB(data_rep)
    result = run_experiment(two_level_kl, data_rep, runs, T, 0)
    result_string['2level KLUCB'].append({'result': result})

print(json.dumps(result_string))
