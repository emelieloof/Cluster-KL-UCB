import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from scipy import stats
from scipy.stats import beta
import random

class MAB_ucb:

    def __init__(self, k, c):
        self.steps = 1
        self.counts = np.zeros(k)
        self.k = k
        self.c = c
    
    def initialize(self,k):
      self.steps = 1
      self.counts = np.zeros(k)
        
    def select_arm(self, k_reward, counts):
        if self.steps <= self.k:
            arm = self.steps - 1
        else:
            arm = np.argmax(k_reward/counts + self.c * np.sqrt(
                (np.log(self.steps)) / self.counts))
        return arm
    
    def update(self,arm):
        self.steps += 1
        self.counts[arm] += 1
        
        
        
        
class MAB_thompson:
  
  def __init__(self, k):
    self.counts = np.zeros(k)
    self.S = np.ones(k)
    self.F = np.ones(k)
    return

  def initialize(self, k):
        self.counts = np.zeros(k)
        self.a = np.ones(k)
        self.b = np.ones(k)
        return

  def select_arm(self):
    parameters = zip(self.S, self.F)
    draws = [beta.rvs(i[0], i[1], size = 1) for i in parameters]
    return draws.index(max(draws))

  def update(self, arm, reward):
      self.counts[arm] += 1
      self.S[arm] += reward
      self.F[arm] += (1-reward)
      return



class BernoulliArm():
    def __init__(self, p):
        self.p = p
    
    def draw(self):
        if random.random() > self.p:
            return 0
        else:
            return 1

class BernoulliBandit:
    def __init__(self, theta):
        self.k = len(theta)
        self.sub_optimality = np.max(theta) - theta
        self.arms = list(map(lambda m: BernoulliArm(m), mu))

    def draw(self, i):
        return self.arms[i].draw()

    def regret(self, i):
        return self.sub_optimality[i]

        

def run_algo_UCB(algo, bandit, rewards, N, runs):
  cumulative_rewards = np.zeros(runs)
  cumulative_regret = np.zeros(runs)
  for sim in range(N):
    algo.initialize(bandit.k)
    new_c_rewards = np.zeros(runs)
    new_c_regrets = np.zeros(runs)
    counts = np.zeros(bandit.k)
    for t in range(runs):
      chosen_arm = algo.select_arm(rewards, counts)
      counts[chosen_arm] += 1
      reward = bandit.draw(chosen_arm)
      regret = bandit.regret(chosen_arm)
      if t == 0:
        new_c_rewards[t] = reward
        new_c_regrets[t] = regret
      else:
        new_c_rewards[t] = new_c_rewards[t-1] + reward
        new_c_regrets[t] = new_c_regrets[t-1] + regret
      rewards[chosen_arm] += reward
      algo.update(chosen_arm)
        
    cumulative_rewards += new_c_rewards
    cumulative_regret += new_c_regrets
    
  return [cumulative_rewards/N, cumulative_regret/N]
  
  
  
def run_algo_TS(algo, bandit, N, runs):
    cumulative_rewards = np.zeros(runs)
    cumulative_regret = np.zeros(runs)
    for sim in range(N):
        algo.initialize(bandit.k)
        new_c_rewards = np.zeros(runs)
        new_c_regrets = np.zeros(runs)
        for t in range(runs):
           chosen_arm = algo.select_arm()
           reward = bandit.draw(chosen_arm)
           regret = bandit.regret(chosen_arm)
           if t == 0:
             new_c_rewards[t] = reward
             new_c_regrets[t] = regret
           else:
             new_c_rewards[t] = new_c_rewards[t-1] + reward
             new_c_regrets[t] = new_c_regrets[t-1] + regret
           algo.update(chosen_arm, reward)

        cumulative_rewards += new_c_rewards
        cumulative_regret += new_c_regrets
    
    return [cumulative_rewards/N, cumulative_regret/N]
    
    
    
N = 10
runs = 1000

mu = [0.3, 0.2, 0.9, 0.15, 0.35]
#arms = list(map(lambda m: BernoulliArm(m), mu))
bandit = BernoulliBandit(mu)

rewards_UCB = np.zeros(len(mu))
c_ucb = 0.5

UCB = MAB_ucb(len(mu), c_ucb)
results_UCB = run_algo_UCB(UCB, bandit, rewards_UCB, N, runs)

TS = MAB_thompson(len(mu))
results_TS = run_algo_TS(TS, bandit, N, runs)

cumulative_regret_UCB = results_UCB[1]
cumulative_regret_TS = results_TS[1]


plt.figure(figsize=(12,8))
plt.plot(cumulative_regret_UCB, label = 'UCB')
plt.plot(cumulative_regret_TS, label = 'TS')
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Trials")
plt.ylabel("Cumulative Regret")
plt.title("Average Cumulative Regret " 
          + str(N) + " Runs")
#plt.show()
plt.savefig('mab_test.pdf')
