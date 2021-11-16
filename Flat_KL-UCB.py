import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math as m


class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def draw(self):
        if rnd.random() > self.p:
            return 0
        else:
            return 1


class BernoulliBandit:
    def __init__(self, theta):
        suboptimality = []
        for i in range(len(theta)):
            suboptimality.append([x1 - x2 for (x1, x2) in zip([max(max(theta))] * len(theta[i]), theta[i])])
        self.sub_optimality = suboptimality
        arms = []
        for i in range(len(theta)):
            arms.append(list(map(lambda m: BernoulliArm(m), theta[i])))
        self.arms = arms

    def draw(self, cluster, arm):
        return self.arms[cluster][arm].draw()

    def regret(self, cluster, arm):
        return self.sub_optimality[cluster][arm]


def kl(p, q):
    epsilon = 0.0001
    if p == 0:
        p += epsilon
    elif p == 1:
        p -= epsilon
    elif q == 0:
        q += epsilon
    elif q == 1:
        q -= epsilon
    return (p * m.log2(p / q)) + ((1 - p) * m.log2((1 - p) / (1 - q)))


class FlatKLUCB:

    def __init__(self, data):
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
            cluster = self.cluster_belongings[t]
            arm = self.arm_belongings[t]
        else:
            means = self.get_means()
            leader_c, leader_a = means.index(max(means)), np.argmax(max(means))
            bounds = self.get_bounds(means, t)
            bounds_c, bounds_a = bounds.index(max(bounds)), np.argmax(max(bounds))
            if leader_c == bounds_c and leader_a == bounds_a:
                cluster, arm = leader_c, leader_a
            else:
                cluster, arm = bounds_c, np.argmin(self.counts[bounds_c])
        return cluster, arm

    def get_means(self):
        means = []
        for i in range(len(self.counts)):
            curr = []
            for j in range(len(self.counts[i])):
                curr.append(self.rewards[i][j] / self.counts[i][j])
            means.append(curr)
        return means

    def get_bounds(self, means, t):
        Q = []
        for i in range(len(means)):
            Q.append([0] * len(means[i]))

        epsilon = 0.001
        max_cluster = means.index(max(means))
        for i in range(len(means)):
            for j in range(len(means[i])):
                q = np.linspace(means[i][j] + epsilon, 1 - epsilon, 200)
                low = 0
                high = len(q) - 1
                while high > low:
                    mid = round((low + high) / 2)
                    prod = np.sum((self.counts[i][j]) * (kl(means[i][j], q[mid])))
                    if prod < np.log(t):
                        low = mid + 1
                    else:
                        high = mid - 1
                Q[i][j] = q[low]

        max_q = max(Q[max_cluster])
        for i in range(len(means)):
            if not i == max_cluster:
                if np.any(np.array(Q[i]) > max_q):
                    max_index = np.argmax(Q[i])
                    K_c = self.get_kl(i, max_index, means, max_cluster)
                    prod = np.sum((self.counts[i][max_index]) * (kl(means[i][max_index], Q[i][max_index])))
                    if not prod + K_c < np.log(t):
                        Q[i][max_index] = min(means[max_cluster])
        return Q

    def get_kl(self, i, j, means, max_cluster):
        sum = 0
        for k in range(len(self.counts[i])):
            sum += self.counts[i][k] * kl(means[i][k], max(means[max_cluster]))
        sum -= self.counts[i][j] * kl(means[i][j], max(means[max_cluster]))
        return sum

    def update(self, arm, cluster, reward, regret):
        self.counts[cluster][arm] += 1
        self.rewards[cluster][arm] += reward
        self.regret.append(regret)


########### TO RUN ##############

T = 10000
runs = 10

data = [[0.2, 0.3, 0.35, 0.38, 0.28], [0.3, 0.41, 0.36, 0.26, 0.38], [0.72, 0.81, 0.78, 0.69, 0.75],[0.33, 0.27, 0.41, 0.39, 0.37], [0.32, 0.17, 0.28, 0.36, 0.28]]
#data = [[0.2, 0.3, 0.35], [0.7, 0.8, 0.6], [0.32, 0.17, 0.28]]
bandit = BernoulliBandit(data)

data_rep = []
for i in range(len(data)):
    data_rep.append([0] * len(data[i]))

Flat_KLUCB = FlatKLUCB(data_rep)
c_regret_flat = []
for i in range(runs):
    Flat_KLUCB.initialize(data_rep)
    for t in range(T):
        cluster, arm = Flat_KLUCB.select_arm(t)
        reward = bandit.draw(cluster, arm)
        regret = bandit.regret(cluster, arm)
        Flat_KLUCB.update(arm, cluster, reward, regret)

    c_regret_flat.append(np.cumsum(Flat_KLUCB.regret))


c_regret_flat = [sum(x) for x in zip(*c_regret_flat)]
c_regret_flat = [c_regret_flat[i] / runs for i in range(len(c_regret_flat))]

########## PLOTTING #######

plt.figure(figsize=(12, 8))
plt.plot(c_regret_flat, label='FLat-KLUCB')

plt.legend(loc='upper left')
plt.xlabel("Trials")
plt.ylabel("Cumulative Regret")
plt.title("Average Cumulative Regret over "
          + str(runs) + " Runs")
plt.show()

########
