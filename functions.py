import random as rnd
import math as m
import numpy as np


class BernoulliArm:  # A class to create an arm and draw a reward from its Bernoulli distribution
    def __init__(self, p):
        self.p = p

    def draw(self):
        if rnd.random() > self.p:
            return 0
        else:
            return 1


class BernoulliBandit:  # A class which creates an actual bandit out of a Bernoulli arm,
    def __init__(self, theta):
        subopt = []
        for i in range(len(theta)):
            subopt.append([x1 - x2 for (x1, x2) in zip([max(max(theta))] * len(theta[i]), theta[i])])
        self.sub_opt = subopt
        arms = []
        for i in range(len(theta)):
            arms.append(list(map(lambda m: BernoulliArm(m), theta[i])))
        self.arms = arms

    def draw(self, cluster, arm):
        return self.arms[cluster][arm].draw()

    def regret(self, cluster, arm):
        return self.sub_opt[cluster][arm]


def kl(p, q):  # function to get the kl-divergence between p and q
    epsilon = 0.0001
    if p == 0:
        p += epsilon
    elif p == 1:
        p -= epsilon
    lif q == 0:
        q += epsilon
    elif q == 1:
        q -= epsilon
    return (p * m.log(p / q)) + ((1 - p) * m.log((1 - p) / (1 - q)))


def run_experiment(algo, data_rep, runs, T, bandit, order):
    c_regret = []
    bars = []
    for i in range(runs):
        algo.initialize(data_rep)
        algo_list = []
        x = []
        for t in range(T):
            cluster, arm = algo.select_arm(t)
            reward = bandit.draw(cluster, arm)
            regret = bandit.regret(cluster, arm)
            algo.update(arm, cluster, reward, regret)
            p = t + 1
            if p % 2000 == 0:
                algo_list.append(np.cumsum(algo.regret)[t - ((1 + order) * 100)])
                x.append(t)

        bars.append(algo_list)
        c_regret.append(np.cumsum(algo.regret))
        print(i)

    c_regret = [sum(x) for x in zip(*c_regret)]
    c_regret = [c_regret[i] / runs for i in range(len(c_regret))]

    error = []
    for i in range(len(bars[0])):
        error.append([np.std([bars[j][i] for j in range(len(bars))]) / runs,
                      np.std([bars[j][i] for j in range(len(bars))]) / runs])

    errors = [[error[i][0] for i in range(len(error))],
                    [error[i][1] for i in range(len(error))]]

    return c_regret, x, errors
