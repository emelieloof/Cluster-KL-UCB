
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import random


class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0
        else:
            return 1


class BernoulliBandit:
    def __init__(self, theta):
        self.k_c = theta.shape[0]
        self.k_a = theta.shape[1]
        self.sub_optimality = np.max(theta) - theta
        self.arms = list(map(lambda m: BernoulliArm(m), theta.reshape((1, self.k_c * self.k_a))[0]))
        self.arms = np.reshape(self.arms, (self.k_c, self.k_a))

    def draw(self, cluster, arm):
        return self.arms[cluster, arm].draw()

    def regret(self, cluster, arm):
        return self.sub_optimality[cluster, arm]


def kl(p, q):
    if p == 0:
        return np.log(1 / (1 - q))
    if p == 1:
        return np.log(p / q)
    else:
        return ((p * np.log(p / q)) + ((1 - p) * np.log((1 - p) / (1 - q))))


def get_bounds(counts, rewards, k, t):
    Q = np.zeros(k)
    epsilon = 0.001
    for i in range(k):
        mean_i = rewards[i] / counts[i]
        q = np.linspace(mean_i + epsilon, 1 - epsilon, 200)
        low = 0
        high = len(q) - 1
        while high > low:
            mid = round((low + high) / 2)
            prod = np.sum(counts[i] * kl(mean_i, q[mid]))
            if prod < np.log(t):
                low = mid + 1
            else:
                high = mid - 1

        Q[i] = q[low]
    return Q


def generate_clusters(arms, clusters, optimal):
    epsilon = 0.0001
    w = 0.1
    d = 0.1
    data = np.zeros((clusters, arms))
    data[0, 0] = optimal
    data[0, -1] = optimal - w
    data[0, 1:arms - 1] = np.random.uniform(optimal - w + epsilon, optimal - epsilon, arms - 2)
    random.shuffle(data[0])
    for i in range(clusters - 1):
        data[i + 1, :] = np.random.uniform(optimal - 2 * w - d, optimal - w - d, arms)
        random.shuffle(data[i + 1])
    return data


def two_level(algo, bandit, t):
    cluster = algo.select_cluster(t)
    arm = algo.select_arm(cluster, t)

    reward = bandit.draw(cluster, arm)
    regret = bandit.regret(cluster, arm)

    algo.update(arm, cluster, reward, regret, t)
    return


############# TS ##############

class TwolevelTS:

    def __init__(self, k_a, k_c, T):
        self.counts_a = np.zeros((k_c, k_a))
        self.counts_c = np.zeros(k_c)
        self.S_a = np.ones((k_c, k_a))
        self.F_a = np.ones((k_c, k_a))
        self.S_c = np.ones(k_c)
        self.F_c = np.ones(k_c)
        self.c_regret = np.zeros(T)
        return

    def initialize(self, k_a, k_c, T):
        self.counts_a = np.zeros((k_c, k_a))
        self.counts_c = np.zeros(k_c)
        self.S_a = np.ones((k_c, k_a))
        self.F_a = np.ones((k_c, k_a))
        self.S_c = np.ones(k_c)
        self.F_c = np.ones(k_c)
        self.c_regret = np.zeros(T)
        return

    def select_cluster(self, t):
        parameters = zip(self.S_c, self.F_c)
        draws = [beta.rvs(i[0], i[1], size=1) for i in parameters]
        return draws.index(max(draws))

    def select_arm(self, cluster, t):
        parameters = zip(self.S_a[cluster, :], self.F_a[cluster, :])
        draws = [beta.rvs(i[0], i[1], size=1) for i in parameters]
        return draws.index(max(draws))

    def update(self, arm, cluster, reward, regret, t):
        self.counts_a[cluster, arm] += 1
        self.counts_c[cluster] += 1
        self.S_c[cluster] += reward
        self.F_c[cluster] += (1 - reward)
        self.S_a[cluster, arm] += reward
        self.F_a[cluster, arm] += (1 - reward)
        if t == 0:
            self.c_regret[0] = regret
        else:
            self.c_regret[t] = self.c_regret[t - 1] + regret
        return


############KLUCB#################

class TwolevelKLUCB:

    def __init__(self, k, c, which_type, T):
        self.steps = 1
        self.counts = np.zeros((c, k))
        self.k = k
        self.c = c
        self.rewards = np.zeros((c, k))
        self.c_regret = np.zeros(T)
        self.which_type = which_type

    def initialize(self, T):
        self.steps = 1
        self.counts = np.zeros((self.c, self.k))
        self.rewards = np.zeros((self.c, self.k))
        self.c_regret = np.zeros(T)

    def get_cluster_representations(self):
        cluster_values = np.zeros(self.c)
        for i in range(self.c):
            if self.which_type == 'min':
                cluster_values[i] = np.min(self.rewards[i, :])
            elif self.which_type == 'max':
                cluster_values[i] = np.max(self.rewards[i, :])
            elif self.which_type == 'avg':
                cluster_values[i] = np.sum(self.rewards[i, :]) / self.k
        test = cluster_values
        return cluster_values

    def get_cluster_counts(self):
        cluster_counts = np.zeros(self.c)
        for i in range(self.c):
            cluster_counts[i] = np.sum(self.counts[i, :])
        test = cluster_counts
        return cluster_counts

    def select_cluster(self, t):
        if t < self.k * self.c:
            cluster = int(np.floor(t / self.k))
        else:
            cluster_rep = self.get_cluster_representations()
            cluster_counts = self.get_cluster_counts()
            bounds = get_bounds(cluster_counts, cluster_rep, self.c, t)
            cluster = np.argmax(bounds)
        test = cluster
        return cluster

    def select_arm(self, cluster, t):
        if t < self.k * self.c:
            arm = t % self.k
        else:
            bounds = get_bounds(self.counts[cluster, :], self.rewards[cluster, :], self.k, t)
            arm = np.argmax(bounds)
        test = arm
        return arm

    def update(self, arm, cluster, reward, regret, t):
        self.steps += 1
        self.counts[cluster, arm] += 1
        self.rewards[cluster, arm] += reward
        if t == 0:
            self.c_regret[t] = regret
        else:
            self.c_regret[t] = self.c_regret[t - 1] + regret
        return


########### TO RUN ##############

T = 1000
runs = 5

data = generate_clusters(10, 10, optimal=0.6)
# data = np.array([[0.2, 0.3, 0.35], [0.7, 0.6, 0.65]])
bandit = BernoulliBandit(data)
TS = TwolevelTS(data.shape[1], data.shape[0], T)
KLUCB_min = TwolevelKLUCB(data.shape[1], data.shape[0], 'min', T)
KLUCB_max = TwolevelKLUCB(data.shape[1], data.shape[0], 'max', T)
KLUCB_avg = TwolevelKLUCB(data.shape[1], data.shape[0], 'avg', T)

c_regret_TS = np.zeros(T)
c_regret_KLUCBmin = np.zeros(T)
c_regret_KLUCBmax = np.zeros(T)
c_regret_KLUCBavg = np.zeros(T)

for i in range(runs):
    TS.initialize(data.shape[1], data.shape[0], T)
    KLUCB_min.initialize(T)
    KLUCB_max.initialize(T)
    KLUCB_avg.initialize(T)

    for t in range(T):
        two_level(TS, bandit, t)
        two_level(KLUCB_min, bandit, t)
        two_level(KLUCB_max, bandit, t)
        two_level(KLUCB_avg, bandit, t)

    c_regret_TS += TS.c_regret / runs
    c_regret_KLUCBmin += KLUCB_min.c_regret / runs
    c_regret_KLUCBmax += KLUCB_max.c_regret / runs
    c_regret_KLUCBavg += KLUCB_avg.c_regret / runs

########## PLOTTING #######

plt.figure(figsize=(12, 8))
plt.plot(c_regret_TS, label='2-level TS')
plt.plot(c_regret_KLUCBmin, label='2-level KL-UCB-min')
plt.plot(c_regret_KLUCBmax, label='2-level KL-UCB-max')
plt.plot(c_regret_KLUCBavg, label='2-level KL-UCB-avg')
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Trials")
plt.ylabel("Cumulative Regret")
plt.title("Average Cumulative Regret over "
          + str(runs) + " Runs")
plt.show()

########
