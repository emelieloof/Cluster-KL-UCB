import numpy as np
import matplotlib.pyplot as plt
import random as rnd


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
            suboptimality.append([x1 - x2 for (x1, x2) in zip([max(max(theta))]*len(theta[i]), theta[i])])
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
    if p == 0:
        return np.log(1 / (1 - q))
    if p == 1:
        return np.log(p / q)
    else:
        return ((p * np.log(p / q)) + ((1 - p) * np.log((1 - p) / (1 - q))))


def get_means(rewards, counts):
    means = []
    for i in range(len(rewards)):
        to_append = []
        for j in range(len(rewards[i])):
            to_append.append(rewards[i][j]/counts[i][j])
        means.append(to_append)
    return means


def get_bounds(counts, rewards, cluster_belongings, k, t):
    Q = []
    for i in range(len(counts)):
        Q.append([0]*len(counts[i]))

    epsilon = 0.001
    means = get_means(rewards,counts)
    max_cluster = means.index(max(means))
    for i in range(len(counts)):
        for j in range(len(counts[i])):
            q = np.linspace(means[i][j] + epsilon, 1 - epsilon, 200)
            low = 0
            high = len(q) - 1
            while high > low:
                mid = round((low + high) / 2)
                prod = np.sum((counts[i][j]) * (kl(means[i][j], q[mid])))
                if prod < np.log(t):
                   low = mid + 1
                else:
                    high = mid - 1
            current_q = q[low]
            current_cluster = cluster_belongings[i]
            if not i == max_cluster:
                if (current_q < np.max(means[max_cluster]) and current_q > np.min(means[max_cluster])):
                    current_q = np.min(means[max_cluster])
                elif current_q > np.max(means[max_cluster]): #you are here
                    cluster_sum = sum(counts[current_cluster])-counts[i][j]
                    C = cluster_sum*get_kl(means[current_cluster], np.max(means[max_cluster]))
                    if not prod + C < np.log(t):
                        current_q = np.min(means[max_cluster])
            Q[i][j] = current_q
    return Q

def get_kl(means, max_mean):
    sum = 0
    for i in range(len(means)):
        sum += kl(means[i], max_mean)
    return sum

def generate_clusters(arms, clusters, optimal):
    epsilon = 0.0001
    w = 0.1
    d = 0.1
    data = []
    optimal_cluster = list(np.random.uniform(optimal - w + epsilon, optimal - epsilon, arms))
    index = rnd.randint(0,arms-1)
    optimal_cluster[index] = optimal
    data.append(optimal_cluster)
    for i in range(clusters - 1):
        data.append(list(np.random.uniform(optimal - 2 * w - d, optimal - w - d, arms)))
    return data


############FLAT-KLUCB#################

class FlatKLUCB:

    def __init__(self, data, T):
        self.steps = 1
        counts = []
        rewards = []
        for i in range(len(data)):
            counts.append([0]*len(data[i]))
            rewards.append([0]*len(data[i]))
        self.counts = counts
        self.rewards = rewards
        self.c_regret = [0]*T

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


    def initialize(self, data, T):
        self.steps = 1
        counts = []
        rewards = []
        for i in range(len(data)):
            counts.append([0] * len(data[i]))
            rewards.append([0] * len(data[i]))
        self.counts = counts
        self.rewards = rewards
        self.c_regret = [0]*T


    def select_arm(self, t):
        if t < np.sum([len(i) for i in self.rewards]):
            cluster = self.cluster_belongings[t]
            arm = self.arm_belongings[t]
        else:
            bounds = get_bounds(self.counts, self.rewards,self.cluster_belongings, len(self.rewards), t)
            cluster = bounds.index(max(bounds))
            arm = bounds[cluster].index(max(bounds[cluster]))
        return cluster, arm

    def update(self, arm, cluster, reward, regret, t):
        self.steps += 1
        self.counts[cluster][arm] += 1
        self.rewards[cluster][arm] += reward
        if t == 0:
            self.c_regret[t] = regret
        else:
            self.c_regret[t] = self.c_regret[t - 1] + regret
        return


########### TO RUN ##############

T = 5000
runs = 2


data = [[0.2, 0.3, 0.35], [0.7, 0.8, 0.6]]
bandit = BernoulliBandit(data)

data_rep = []
for i in range(len(data)):
    data_rep.append([0]*len(data[i]))

Flat_KLUCB = FlatKLUCB(data_rep,T)

c_regret = [0]*T


for i in range(runs):
    Flat_KLUCB.initialize(data_rep, T)


    for t in range(T):
        cluster, arm = Flat_KLUCB.select_arm(t)
        reward = bandit.draw(cluster, arm)
        regret = bandit.regret(cluster, arm)

        Flat_KLUCB.update(arm, cluster, reward, regret, t)

        c_regret[t] += Flat_KLUCB.c_regret[t]




c_regret = [x / runs for x in c_regret]



########## PLOTTING #######

plt.figure(figsize=(12, 8))
plt.plot(c_regret, label='FLat-KLUCB')

plt.legend(loc='upper left')
plt.xlabel("Trials")
plt.ylabel("Cumulative Regret")
plt.title("Average Cumulative Regret over "
          + str(runs) + " Runs")
plt.show()

########
