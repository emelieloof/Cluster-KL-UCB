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
    
    
########## 2-level Thompson Sampling ##########

class MAB_twolevel_thompson:
  
  def __init__(self, k_a, k_c ):
    self.counts_a = np.zeros((k_c,k_a))
    self.counts_c = np.zeros(k_c)
    self.S_a = np.ones((k_c,k_a))
    self.F_a = np.ones((k_c,k_a))
    self.S_c = np.ones(k_c)
    self.F_c = np.ones(k_c)
    return

  def initialize(self, k_a, k_c):
    self.counts_a = np.zeros((k_c,k_a))
    self.counts_c = np.zeros(k_c)
    self.S_a = np.ones((k_c,k_a))
    self.F_a = np.ones((k_c,k_a))
    self.S_c = np.ones(k_c)
    self.F_c = np.ones(k_c)
    return

  def select_cluster(self):
    parameters = zip(self.S_c, self.F_c)
    draws = [beta.rvs(i[0], i[1], size = 1) for i in parameters]
    return draws.index(max(draws))

  def select_arm(self, cluster):
    parameters = zip(self.S_a[cluster,:], self.F_a[cluster,:])
    draws = [beta.rvs(i[0], i[1], size = 1) for i in parameters]
    return draws.index(max(draws))

  def update(self, arm, cluster, reward):
      self.counts_a[cluster,arm] += 1
      self.counts_c[cluster] += 1
      self.S_c[cluster] += reward
      self.F_c[cluster] += (1-reward)
      self.S_a[cluster,arm] += reward
      self.F_a[cluster,arm] += (1-reward)
      
      return
      
def run_algo_TS(algo, bandit, N, runs):
    cumulative_rewards = np.zeros(runs)
    cumulative_regret = np.zeros(runs)
    for sim in range(N):
        algo.initialize(bandit.k_a, bandit.k_c) 
        new_c_rewards = np.zeros(runs)
        new_c_regrets = np.zeros(runs)
        for t in range(runs):
          chosen_cluster = algo.select_cluster()
          chosen_arm = algo.select_arm(chosen_cluster)
          reward = bandit.draw(chosen_cluster, chosen_arm)
          regret = bandit.regret(chosen_cluster, chosen_arm)
          if t == 0:
            new_c_rewards[t] = reward
            new_c_regrets[t] = regret
          else:
            new_c_rewards[t] = new_c_rewards[t-1] + reward
            new_c_regrets[t] = new_c_regrets[t-1] + regret
          algo.update(chosen_arm, chosen_cluster, reward)

        cumulative_rewards += new_c_rewards
        cumulative_regret += new_c_regrets
    
    return [cumulative_rewards/N, cumulative_regret/N]
    
    
########## 2-level KL-UCB ##########

class MAB_cluster_KL_UCB:

    def __init__(self, k, c):
        self.steps = 1
        self.counts_arms = np.zeros(k)
        self.counts_cluster = np.zeros(c)
        self.k = k
        self.c = c
    
    def initialize(self,k):
      self.steps = 1
      self.counts = np.zeros(k)
    
    def select_cluster(self,rewards, counts, t):
      bounds = get_bounds(counts, rewards, self.c,t)
      cluster = np.argmax(bounds)
      return cluster
        
    def select_arm(self, rewards, counts, t):
        bounds = get_bounds(counts, rewards, len(rewards) , t)
        arm = np.argmax(bounds)
        return arm
      
    def update(self,arm):
        self.steps += 1
        self.counts[arm] += 1
        
def get_cluster_rewards(rewards, c, which_type):
  cluster_values = np.zeros(c)
  for i in range(c):
    if which_type == 'min':
      cluster_values[i] = np.min(rewards[i,:])
    elif which_type == 'max':
      cluster_values[i] = np.max(rewards[i,:])
    elif which_type == 'avg':
      cluster_values[i] = np.sum(rewards[i,:])/np.int(np.shape(rewards)[1])
  return cluster_values
  
  
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
      prod = np.sum(counts*KL(mean_i,q[mid]))
      if prod < np.log(t):
        low = mid + 1
      else:
        high = mid - 1
      
    Q[i] = q[low]
  return Q
      

def run_algo_cluster_KL_UCB(algo, bandit, rewards, N, runs, which_type ):
  cumulative_rewards = np.zeros(runs)
  cumulative_regret = np.zeros(runs)
  arm_rewards = np.reshape(rewards, (bandit.k_c,bandit.k_a))
  for sim in range(N):
    algo.initialize(bandit.k_a*bandit.k_c)
    new_c_rewards = np.zeros(runs)
    new_c_regrets = np.zeros(runs)
    counts_cluster = np.zeros(bandit.k_c)
    counts_arms = np.zeros(bandit.k_a*bandit.k_c)
    counts_arms = np.reshape(counts_arms, (bandit.k_c,bandit.k_a))
    for t in range(runs):
      cluster_rewards = get_cluster_rewards(arm_rewards,bandit.k_c, which_type)
      c = 0
      if t < bandit.k_a*bandit.k_c:
        chosen_arm = t%bandit.k_a
        chosen_cluster = c
        counts_arms[chosen_cluster,chosen_arm] += 1
        counts_cluster[c] += 1
        if c%bandit.k_a == 0:
          c += 1
      else:
        chosen_cluster = algo.select_cluster(cluster_rewards, counts_cluster, t)
        chosen_arm = algo.select_arm(arm_rewards[chosen_cluster,:], counts_arms[chosen_cluster,:], t)
        counts_arms[chosen_cluster,chosen_arm] += 1
        counts_cluster[chosen_cluster] += 1
      reward = bandit.draw(math.floor(chosen_arm/bandit.k_a), np.int(chosen_arm%bandit.k_a ))
      regret = bandit.regret(math.floor(chosen_arm/bandit.k_a), np.int(chosen_arm%bandit.k_a ))
      if t == 0:
        new_c_rewards[t] = reward
        new_c_regrets[t] = regret
      else:
        new_c_rewards[t] = new_c_rewards[t-1] + reward
        new_c_regrets[t] = new_c_regrets[t-1] + regret
      arm_rewards[chosen_cluster,chosen_arm] += reward
      algo.update(chosen_arm)
        
    cumulative_rewards += new_c_rewards
    cumulative_regret += new_c_regrets
    
  return [cumulative_rewards/N, cumulative_regret/N]
  
 ########## Code to run ##########
 
 def generate_clusters(arms, clusters, optimal ):
  epsilon = 0.0001
  w = 0.1
  d = 0.1
  data = np.zeros((clusters, arms))
  data[0,0] = optimal
  data[0,-1] = optimal-w
  data [0,1:arms-1] = np.random.uniform(optimal-w+epsilon,optimal-epsilon,arms-2)
  for i in range(clusters-1):
    data [i+1,:] = np.random.uniform(optimal-2*w-d,optimal-w-d,arms)
  return data


#data = np.array([[0.75, 0.6, 0.55], [0.5, 0.3, 0.2], [0.15,0.18, 0.1]]) 
data = generate_clusters(10,10,optimal=0.6)

N = 10
runs = 10000

bandit = BernoulliBandit(data)


TS = MAB_twolevel_thompson(data.shape[1], data.shape[0])
results_TS = run_algo_TS(TS, bandit, N, runs)

rewards_UCB_min = np.zeros(data.shape[0]*data.shape[1])
UCB_min = MAB_cluster_KL_UCB(data.shape[0]*data.shape[1],data.shape[0])
results_UCB_min = run_algo_cluster_KL_UCB(UCB_min, bandit, rewards_UCB_min, N, runs,'min')

rewards_UCB_max = np.zeros(data.shape[0]*data.shape[1])
UCB_max = MAB_cluster_KL_UCB(data.shape[0]*data.shape[1],data.shape[0])
results_UCB_max = run_algo_cluster_KL_UCB(UCB_max, bandit, rewards_UCB_max, N, runs,'max')

rewards_UCB_avg = np.zeros(data.shape[0]*data.shape[1])
UCB_avg = MAB_cluster_KL_UCB(data.shape[0]*data.shape[1],data.shape[0])
results_UCB_avg = run_algo_cluster_KL_UCB(UCB_avg, bandit, rewards_UCB_avg, N, runs,'avg')

cumulative_regret_TS = results_TS[1]
cumulative_regret_KL_UCB_min = results_UCB_min[1]
cumulative_regret_KL_UCB_max = results_UCB_max[1]
cumulative_regret_KL_UCB_avg = results_UCB_avg[1]


plt.figure(figsize=(12,8))
#plt.plot(cumulative_regret_KL_UCB, label = 'KL-UCB')
plt.plot(cumulative_regret_TS, label = 'TS')
plt.plot(cumulative_regret_KL_UCB_min, label = '2-level KL-UCB-min')
plt.plot(cumulative_regret_KL_UCB_max, label = '2-level KL-UCB-max')
plt.plot(cumulative_regret_KL_UCB_avg, label = '2-level KL-UCB-avg')
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Trials")
plt.ylabel("Cumulative Regret")
plt.title("Average Cumulative Regret " 
          + str(N) + " Runs")
plt.show()
