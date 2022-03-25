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
    
    def get_bounds(self, means, t): #Function to get the Q values of each arm
        if not self.clustering: # This is used for ordinary KL-UCB
            Q = []
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
                            prod = np.sum((self.counts[i][j]) * (fun.kl(means[i][j], q[mid])))
                            if prod < np.log(t):
                                low = mid + 1
                            else:
                                high = mid - 1
                        Q[i][j] = q[low]
            return Q

        else: # Here's for our algo
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
                            if not i == max_mean_index[0]: #Calculating penalty term if we're not in the optimal cluster
                                limit = self.get_limit([i, j], max_mean_index, q[mid], means)
                            else: #Same as ordinary KL-UCB if we're in the optimal cluster
                                limit = self.counts[i][j] * fun.kl(means[i][j], q[mid])
                            if limit < np.log(t):
                                low = mid + 1
                            else:
                                high = mid - 1
                        Q[i][j] = q[low]
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
        new_list = self.reshape(l)
        max = np.max(new_list)
        if isq: #If the optimal cluster is optimal we want to return it (this prevents from returning just the first optimal cluster if several are equal)
            for i in range(len(l)):
                for j in range(len(l[i])):
                    if l[i][j] == max and i == optimal_cluster:
                        return i, j
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
