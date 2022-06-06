import random as rnd
import numpy as np


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
