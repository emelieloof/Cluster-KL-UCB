import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from scipy import stats
from scipy.stats import beta
import random
import math

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
        self.k_c = theta.shape[0]
        self.k_a = theta.shape[1]
        self.sub_optimality = np.max(theta) - theta
        self.arms = list(map(lambda m: BernoulliArm(m), theta.reshape((1,self.k_c*self.k_a))[0]))

    def draw(self, cluster, arm):
        return self.arms[arm + cluster*self.k_c].draw()

    def regret(self, cluster, arm):
        return self.sub_optimality[cluster, arm]

def KL(p,q):
  if p == 0:
    return np.log(1/(1-q))
  if p == 1:
    return np.log(p/q)
  else:
    return ((p*np.log(p/q))+((1-p)*np.log((1-p)/(1-q))))

def get_bounds(counts, rewards, k, t):
  Q = np.zeros(k)
  epsilon = 0.001
  for i in range(k):
    mean_i = rewards[i]/counts[i]
    q = np.linspace(mean_i + epsilon,1 - epsilon,200)
    low = 0
    high = len(q)-1
    while high > low:
      mid = round((low+high)/2)
      prod = counts[i]*KL(mean_i,q[mid])
      if prod < np.log(t):
        low = mid + 1
      else:
        high = mid - 1
      
    Q[i] = q[low]
  return Q
    
    
class MAB_KL_UCB:

    def __init__(self, k, c):
        self.steps = 1
        self.counts = np.zeros(k)
        self.k = k
        self.c = c
    
    def initialize(self,k):
      self.steps = 1
      self.counts = np.zeros(k)
        
    def select_arm(self, rewards, counts, k, t):
        if self.steps <= self.k:
            arm = self.steps - 1
        else:
            bounds = get_bounds(counts, rewards, k, t)
            arm = np.argmax(bounds)
        return arm
      
    def update(self,arm):
        self.steps += 1
        self.counts[arm] += 1
      
    
        
        
def run_algo_KL_UCB(algo, bandit, rewards, N, runs, ):
  cumulative_rewards = np.zeros(runs)
  cumulative_regret = np.zeros(runs)
  for sim in range(N):
    algo.initialize(bandit.k_a*bandit.k_c)
    new_c_rewards = np.zeros(runs)
    new_c_regrets = np.zeros(runs)
    counts = np.zeros(bandit.k_a*bandit.k_c)
    for t in range(runs):
      chosen_arm = algo.select_arm(rewards, counts,bandit.k_a*bandit.k_c, t)
      counts[chosen_arm] += 1
      reward = bandit.draw(math.floor(chosen_arm/bandit.k_a), np.int(chosen_arm%bandit.k_a ))
      regret = bandit.regret(math.floor(chosen_arm/bandit.k_a), np.int(chosen_arm%bandit.k_a ))
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
    
  
#data = np.array([[0.75, 0.6, 0.55], [0.5, 0.3, 0.2], [0.15,0.18, 0.1]]) #Strong-dominance
data = np.array([[0.75,0.62, 0.55],[0.58, 0.47, 0.38], [0.41, 0.31, 0.24]]) #overlap
N = 10
runs = 1000

bandit = BernoulliBandit(data)

rewards_UCB = np.zeros(data.shape[0]*data.shape[1])
c_ucb = 5

UCB = MAB_KL_UCB(data.shape[0]*data.shape[1], c_ucb)
results_UCB = run_algo_KL_UCB(UCB, bandit, rewards_UCB, N, runs)

cumulative_regret_KL_UCB = results_UCB[1]


plt.figure(figsize=(12,8))
plt.plot(cumulative_regret_KL_UCB, label = 'KL-UCB')
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Trials")
plt.ylabel("Cumulative Regret")
plt.title("Average Cumulative Regret " 
          + str(N) + " Runs")
plt.show()

    
    