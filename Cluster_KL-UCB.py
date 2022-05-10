import numpy as np
import random as rnd
import json
from joblib import Parallel, delayed


class c_KLUCB:

    def __init__(self, data, overlap):
        counts = []
        rewards = []
        arms = 0
        for i in range(len(data)):
            counts.append([0] * len(data[i]))
            rewards.append([0] * len(data[i]))
            arms += len(data[i])
        self.counts = counts
        self.rewards = rewards
        self.regret = []
        self.num_arms = arms
        self.overlap = overlap

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
        counts = []
        rewards = []
        for i in range(len(data)):
            counts.append([0] * len(data[i]))
            rewards.append([0] * len(data[i]))
        self.counts = counts
        self.rewards = rewards
        self.regret = []

    def select_arm(self, t):
        if t < self.num_arms:
            reward = bandit.draw(self.cluster_belongings[t], self.arm_belongings[t])
            regret = bandit.regret(self.cluster_belongings[t], self.arm_belongings[t])
            self.update(self.arm_belongings[t], self.cluster_belongings[t], reward, regret)
        else:
            means = self.get_means()
            means_opt_index = self.get_max_index(means, False, 0)
            bounds = self.get_bounds(means, t)
            max_q_index = self.get_max_index(bounds, True, means_opt_index[0])
            reward = bandit.draw(max_q_index[0], max_q_index[1])
            regret = bandit.regret(max_q_index[0], max_q_index[1])
            self.update(max_q_index[1], max_q_index[0], reward, regret)

    def get_bounds(self, means, t):
        Q = []
        max_mean_index = self.get_max_index(means, False, 0)
        for i in range(len(means)):
            Q.append([0] * len(means[i]))

        for i in range(len(means)):
            for j in range(len(means[i])):
                if means[i][j] == 1:
                    Q[i][j] = 1
                else:
                    q = np.linspace(means[i][j], 1, 50)
                    low = 0
                    high = len(q) - 1
                    while high > low:
                        mid = round((low + high) / 2)
                        if not i == max_mean_index[0]:
                            limit = self.get_limit([i, j], max_mean_index, q[mid], means)
                        else:
                            limit = self.counts[i][j] * kl(means[i][j], q[mid])
                        if limit < np.log(t):
                            low = mid + 1
                        else:
                            high = mid - 1
                    Q[i][j] = q[low]
        return Q

    def get_means(self):
        means = []
        for i in range(len(self.counts)):
            curr = []
            for j in range(len(self.counts[i])):
                curr.append(self.rewards[i][j] / self.counts[i][j])
            means.append(curr)
        return means

    def get_kl(self, i, j, means, max_cluster):
        sum = 0
        for k in range(len(self.counts[i])):
            index = max((max(means[max_cluster]) - self.overlap), means[i][k])
            sum += self.counts[i][k] * kl(means[i][k], index)
        sum -= self.counts[i][j] * kl(means[i][j], max((max(means[max_cluster]) - self.overlap), means[i][j]))
        return sum

    def get_limit(self, index, max_mean_index, q, means):
        kl_sum = self.get_kl(index[0], index[1], means, max_mean_index[0])
        max_q_prod = self.counts[index[0]][index[1]] * kl(means[index[0]][index[1]], q)
        result = kl_sum + max_q_prod
        return result

    def get_max_index(self, l, isq, optimal_cluster):
        new_list = self.reshape(l)
        max = np.max(new_list)
        if isq:
            for i in range(len(l)):
                for j in range(len(l[i])):
                    if l[i][j] == max and i == optimal_cluster:
                        return i, j
        max_index = np.argmax(new_list)
        return self.cluster_belongings[max_index], self.arm_belongings[max_index]

    def reshape(self, list):
        new_list = np.zeros(len(self.arm_belongings))
        index = 0
        for i in range(len(list)):
            for j in range(len(list[i])):
                new_list[index] = list[i][j]
                index += 1
        return new_list

    def update(self, arm, cluster, reward, regret):
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


def one_run(algo, data_rep, T, r):
    algo.initialize(data_rep)
    for t in range(T):
        algo.select_arm(t)
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


def get_overlap(data, t):
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


data1 = [[0.6, 0.8, 0.7], [0.3, 0.5, 0.4]]  # overlap = - 50%
data2 = [[0.65, 0.7, 0.8, 0.6, 0.75], [0.35, 0.4, 0.5, 0.3, 0.45], [0.25, 0.3, 0.4, 0.2, 0.35],
         [0.25, 0.3, 0.4, 0.2, 0.35], [0.25, 0.3, 0.4, 0.2, 0.35]]  # overlap = -50%
data3 = [[0.6, 0.8, 0.7], [0.5, 0.7, 0.6]]  # overlap = 50 %
data4 = [[0.65, 0.7, 0.8, 0.6, 0.75], [0.55, 0.6, 0.7, 0.5, 0.65], [0.25, 0.3, 0.4, 0.2, 0.35],
         [0.25, 0.3, 0.4, 0.2, 0.35], [0.25, 0.3, 0.4, 0.2, 0.35]] # overlap = 50 %
d = [data1, data2, data3, data4]


T = 1000
runs = 50
result_string = {'Clustered KL-UCB': []}

for data in d:
    bandit = BernoulliBandit(data)
    data_rep = []
    for i in range(len(data)):
        data_rep.append([0] * len(data[i]))
    overlap = get_overlap(data)
    c_klucb = c_KLUCB(data_rep, overlap)
    results_c_klucb = run_experiment(c_klucb, data_rep, runs, T, j)

    result_string['Clustered KL-UCB'].append({'result': results_c_klucb})
    
with open('wanted_filename.json', 'w') as f:
    f.truncate(0)
    json.dump(result_string, f)
