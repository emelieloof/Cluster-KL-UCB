class KLUCB:

    def __init__(self, data, clustering, overlap):
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
        self.clustering = clustering # A boolean to indicate if we want to use our algo or ordinary KL-UCB
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

    def select_arm(self, t): # A function which selects which arm we want to play at time t
        if t < self.num_arms:
            cluster = self.cluster_belongings[t]
            arm = self.arm_belongings[t]
        else:
            means = self.get_means()
            bounds = self.get_bounds(means, t)
            max_q_index = self.get_max_index(bounds)
            cluster, arm = max_q_index[0], max_q_index[1]

        return cluster, arm

    def get_means(self): # a function to calculate the current means of each arm
        means = []
        for i in range(len(self.counts)):
            curr = []
            for j in range(len(self.counts[i])):
                curr.append(self.rewards[i][j] / self.counts[i][j])
            means.append(curr)
        return means

    def get_bounds(self, means, t): # this dunction caculates the q-values for each arm
        Q = []
        for i in range(len(means)):
            Q.append([0] * len(means[i]))

        epsilon = 0.001
        for i in range(len(means)):
            for j in range(len(means[i])):
                if means[i][j] == 1: # This condition is nessasary since the interval in which we search the q-value can't be empty
                    Q[i][j] = 1
                else: # find the optimal q with binary search
                    q = np.linspace(means[i][j], 1, 50) 
                    low = 0
                    high = len(q) - 1
                    while high > low:
                        mid = round((low + high) / 2)
                        prod = np.sum((self.counts[i][j]) * (fun.kl(means[i][j], q[mid])))
                        if prod < np.log(t + np.log(t)):
                            low = mid + 1
                        else:
                            high = mid - 1
                    Q[i][j] = q[low]

        if self.clustering: # If we uant to use our algorithm we want to see if the additional conditions hold
            max_mean_index = self.get_max_index(means)
            max_q_index = self.get_max_index(Q)
            if max_mean_index[0] == max_q_index[0]: # If the max q-value belongs to the current optimal cluster we donät need to check additonal conditions
                return Q
            else:
                 for i in range(len(means)):# looping through the clusters
                    if not i == max_mean_index[0] and np.any(Q[i] >= np.max(Q[max_mean_index[0]])):# checking if we're in the optimal cluster and if any of the q-values are greater than the q- value of the optimal arm
                        for j in range(len(means[i])): #looping through the cluster
                            if Q[i][j] >= np.max(Q[max_mean_index[0]]): #checking if the q-value of the current arm is greater than the q-value of the optimal arm 
                                limit = self.get_limit([i, j], max_mean_index, Q, means) # calculating the penalty term
                                if not limit < np.log(t) + np.log(np.log(t)): #checking if condition holds
                                    Q[i][j] = min(means[max_mean_index[0]]) # if it doesn't hold lower the q-value so it doesn't get 
                return Q
        return Q


    def get_kl(self, i, j, means, max_cluster): # function to get the sum of the kl-terms used in the penalty term
        sum = 0
        for k in range(len(self.counts[i])):
            sum += self.counts[i][k] * fun.kl(means[i][k], max(means[max_cluster]) - self.overlap)
        sum -= self.counts[i][j] * fun.kl(means[i][j], max(means[max_cluster]) - self.overlap)
        return sum

    def get_limit(self,max_q_index, max_mean_index, Q, means): # function to get the penalty term
        kl_sum = self.get_kl(max_q_index[0], max_q_index[1], means, max_mean_index[0])
        max_q_prod = self.counts[max_q_index[0]][max_q_index[1]] * \
                     fun.kl(means[max_q_index[0]][max_q_index[1]], Q[max_q_index[0]][max_q_index[1]])
        return kl_sum + max_q_prod

        def get_max_index(self, l, isq, optimal_cluster): # function to get max index of uneven list
        if isq: 
            for i in range(len(l)):
                if not i == optimal_cluster: # Want to make sure we get the optimal cluster as optimal
                    for j in range(len(l[i])):
                        l[i][j] == 0
        new_list = self.reshape(l)
        max_index = np.argmax(new_list)

        return self.cluster_belongings[max_index], self.arm_belongings[max_index]
    def reshape(self, list):# Function to reshape an uneven list
        new_list = np.zeros(len(self.arm_belongings))
        index = 0
        for i in range(len(list)):
            for j in range(len(list[i])):
                new_list[index] = list[i][j]
                index += 1
        return new_list

    def update(self, arm, cluster, reward, regret): #Function to update the algorithm
        self.counts[cluster][arm] += 1
        self.rewards[cluster][arm] += reward
        self.regret.append(regret)
