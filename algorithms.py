import numpy as np
import functions as fun
from random import sample


class TS:

    def __init__(self, data, bandit):
        self.counts = data
        S_a = []
        F_a = []
        for i in range(len(data)):
            S_a.append([1] * len(data[i]))
            F_a.append([1] * len(data[i]))
        self.S_a = S_a
        self.F_a = F_a
        self.regret = []
        self.bandit = bandit
        self.cur_c = 0
        self.cur_a = 0

        cluster_belongings = [0] * np.sum([len(i) for i in data])
        index = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                cluster_belongings[index] = i
                index += 1
        self.cluster_belongings = cluster_belongings

        arm_belongings = [0] * np.sum([len(i) for i in data])
        index = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                arm_belongings[index] = j
                index += 1
        self.arm_belongings = arm_belongings
        return

    def initialize(self, data):
        self.counts = data
        S_a = []
        F_a = []
        for i in range(len(data)):
            S_a.append([1] * len(data[i]))
            F_a.append([1] * len(data[i]))
        self.S_a = S_a
        self.F_a = F_a
        self.regret = []
        return

    def select_arm(self):
        s, f = np.reshape(self.S_a, (1, len(self.cluster_belongings))), np.reshape(self.F_a,
                                                                                   (1, len(self.cluster_belongings)))
        parameters = zip(s[0], f[0])
        draws = [np.random.beta(x1, x2) for (x1, x2) in parameters]
        max_index = draws.index(np.max(draws))
        self.cur_c = self.cluster_belongings[max_index]
        self.cur_a = self.arm_belongings[max_index]
        self.update()

    def update(self):
        reward = self.bandit.draw(self.cur_c, self.cur_a)
        regret = self.bandit.regret(self.cur_c, self.cur_a)
        self.counts[self.cur_c][self.cur_a] += 1
        self.S_a[self.cur_c][self.cur_a] += reward
        self.F_a[self.cur_c][self.cur_a] += (1 - reward)
        self.regret.append(regret)


class TwolevelTS:

    def __init__(self, data, bandit):
        self.counts = data
        S_a = []
        F_a = []
        for i in range(len(data)):
            S_a.append([1] * len(data[i]))
            F_a.append([1] * len(data[i]))
        self.S_a = S_a
        self.F_a = F_a
        self.S_c = [1] * len(data)
        self.F_c = [1] * len(data)
        self.regret = []
        self.bandit = bandit
        self.cur_c = 0
        self.cur_a = 0
        return

    def initialize(self, data):
        self.counts = data
        S_a = []
        F_a = []
        for i in range(len(data)):
            S_a.append([1] * len(data[i]))
            F_a.append([1] * len(data[i]))
        self.S_a = S_a
        self.F_a = F_a
        self.S_c = [1] * len(data)
        self.F_c = [1] * len(data)
        self.regret = []
        return

    def select_cluster(self):
        draws = [np.random.beta(x1, x2) for (x1, x2) in zip(self.S_c, self.F_c)]
        self.cur_c = draws.index(np.max(draws))

    def select_arm(self):
        self.select_cluster()
        parameters = zip(self.S_a[self.cur_c], self.F_a[self.cur_c])
        draws = [np.random.beta(x1, x2) for (x1, x2) in parameters]
        self.cur_a = draws.index(np.max(draws))
        self.update()

    def update(self):
        reward = self.bandit.draw(self.cur_c, self.cur_a)
        regret = self.bandit.regret(self.cur_c, self.cur_a)
        self.counts[self.cur_c][self.cur_a] += 1
        self.S_c[self.cur_c] += reward
        self.F_c[self.cur_c] += (1 - reward)
        self.S_a[self.cur_c][self.cur_a] += reward
        self.F_a[self.cur_c][self.cur_a] += (1 - reward)
        self.regret.append(regret)


class TwolevelKLUCB:

    def __init__(self, data, type, bandit):
        self.steps = 1
        counts = []
        rewards = []
        for i in range(len(data)):
            counts.append([0] * len(data[i]))
            rewards.append([0] * len(data[i]))
        self.counts = counts
        self.rewards = rewards
        self.regret = []
        self.type = type
        self.t = 0
        self.cur_c = 0
        self.cur_a = 0
        self.bandit = bandit

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
        self.steps = 1
        counts = []
        rewards = []
        for i in range(len(data)):
            counts.append([0] * len(data[i]))
            rewards.append([0] * len(data[i]))
        self.counts = counts
        self.rewards = rewards
        self.regret = []
        self.t = 0

    def get_means(self):
        means = []
        for i in range(len(self.counts)):
            curr = []
            for j in range(len(self.counts[i])):
                curr.append(self.rewards[i][j] / self.counts[i][j])
            means.append(curr)
        return means

    def get_cluster_means(self):
        cluster_means = [0] * len(self.rewards)
        means = self.get_means()
        for i in range(len(self.rewards)):
            if self.type == 'min':
                min_index = means[i].index(min(means[i]))
                cluster_means[i] = means[i][min_index]
            elif self.type == 'max':
                max_index = means[i].index(max(means[i]))
                cluster_means[i] = means[i][max_index]
            elif self.type == 'avg':
                cluster_means[i] = np.sum(means[i]) / len(means[i])
        return cluster_means

    def get_bounds(self, means, level, c, t):
        Q = []
        for i in range(len(means)):
            if level == 1:
                full_means = self.get_means()
                if self.type == 'max':
                    max_index = full_means[i].index(max(full_means[i]))
                    counts = self.counts[i][max_index]
                elif self.type == 'min':
                    min_index = full_means[i].index(min(full_means[i]))
                    counts = self.counts[i][min_index]
                else:
                    counts = sum(self.counts[i]) / len(self.counts[i])
            else:
                counts = self.counts[c][i]

            if means[i] == 1:
                Q.append(1)
            else:
                q = np.linspace(means[i], 1, 50)
                low = 0
                high = len(q) - 1
                while high > low:
                    mid = round((low + high) / 2)
                    prod = np.sum(counts * (fun.kl(means[i], q[mid])))
                    if prod < np.log(t):
                        low = mid + 1
                    else:
                        high = mid - 1
                Q.append(q[low])
        return Q

    def select_cluster(self):
        if self.t < np.sum([len(i) for i in self.rewards]):
            self.cur_c = self.cluster_belongings[self.t]
        else:
            means = self.get_cluster_means()
            bounds = self.get_bounds(means, 1, None, self.t)
            self.cur_c = np.argmax(bounds)

    def select_arm(self):
        self.select_cluster()
        if self.t < np.sum([len(i) for i in self.rewards]):
            self.cur_a = self.arm_belongings[self.t]
        else:
            means = self.get_means()
            self.cur_a = np.argmax(self.get_bounds(means[self.cur_c], 2, self.cur_c, self.t))
        self.update()

    def update(self):
        self.steps += 1
        reward = self.bandit.draw(self.cur_c, self.cur_a)
        regret = self.bandit.regret(self.cur_c, self.cur_a)
        self.counts[self.cur_c][self.cur_a] += 1
        self.rewards[self.cur_c][self.cur_a] += reward
        self.regret.append(regret)
        self.t += 1


class KLUCB:

    def __init__(self, data, bandit):
        counts = []
        rewards = []
        for i in range(len(data)):
            counts.append([0] * len(data[i]))
            rewards.append([0] * len(data[i]))
        self.counts = counts
        self.rewards = rewards
        self.regret = []
        self.cur_c = 0
        self.cur_a = 0
        self.t = 0
        self.bandit = bandit

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
        self.t = 0

    def get_means(self):
        means = []
        for i in range(len(self.counts)):
            curr = []
            for j in range(len(self.counts[i])):
                curr.append(self.rewards[i][j] / self.counts[i][j])
            means.append(curr)
        return means

    def select_arm(self):
        if self.t < np.sum([len(i) for i in self.rewards]):
            self.cur_c = self.cluster_belongings[self.t]
            self.cur_a = self.arm_belongings[self.t]
            self.update()
        else:
            means = self.get_means()
            means_opt_index = self.get_max_index(means, False, 0)
            bounds = self.get_bounds(means, self.t)
            max_q_index = self.get_max_index(bounds, True, means_opt_index[0])
            self.cur_c = max_q_index[0]
            self.cur_a = max_q_index[1]
            self.update()

    def get_bounds(self, means, t):
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

    def get_max_index(self, l, isq, optimal_cluster):
        new_list = self.reshape(l)
        max = np.max(new_list)
        if isq:
            for i in range(len(l)):
                for j in range(len(l[i])):
                    if l[i][j] == max and i == optimal_cluster:
                        return i, j
        max_index = np.argmax(new_list)
        return self.cluster_belongings[max_index], self.arm_belongings[max_index]

    def reshape(self, list):
        new_list = np.zeros(len(self.arm_belongings))
        index = 0
        for i in range(len(list)):
            for j in range(len(list[i])):
                new_list[index] = list[i][j]
                index += 1
        return new_list

    def update(self):
        reward = self.bandit.draw(self.cur_c, self.cur_a)
        regret = self.bandit.regret(self.cur_c, self.cur_a)
        self.counts[self.cur_c][self.cur_a] += 1
        self.rewards[self.cur_c][self.cur_a] += reward
        self.regret.append(regret)
        self.t += 1


class c_KLUCB:

    def __init__(self, data, type, overlap, bandit):
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
        self.overlap = overlap
        self.cur_c = 0
        self.cur_a = 0
        self.bandit = bandit
        self.t = 0
        self.type = type

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
        self.t = 0

    def select_arm(self):
        if self.t < self.num_arms:
            self.cur_c = self.cluster_belongings[self.t]
            self.cur_a = self.arm_belongings[self.t]
            self.update()
        else:
            means = self.get_means()
            means_opt_index = self.get_max_index(means, False, 0)
            bounds = self.get_bounds(means, self.t)
            max_q_index = self.get_max_index(bounds, True, means_opt_index[0])
            if self.type == 'min':
                if max_q_index[0] == means_opt_index[0]:
                    self.cur_c = max_q_index[0]
                    self.cur_a = max_q_index[1]
                else:
                    self.cur_c = max_q_index[0]
                    self.cur_a = np.argmin(self.counts[max_q_index[0]])
            elif self.type == 'avg':
                new_bounds = self.get_rand_bounds(bounds, means)
                max_index = self.get_max_index(new_bounds, True, means_opt_index[0])
                self.cur_c = max_index[0]
                self.cur_a = max_index[1]
            else:
                self.cur_c = max_q_index[0]
                self.cur_a = max_q_index[1]
            self.update()

    def get_bounds(self, means, t):
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
                        if not i == max_mean_index[0]:
                            limit = self.get_limit([i, j], max_mean_index, q[mid], means)
                        else:
                            limit = self.counts[i][j] * fun.kl(means[i][j], q[mid])
                        if limit < np.log(t):
                            low = mid + 1
                        else:
                            high = mid - 1
                    Q[i][j] = q[low]
        return Q

    def get_rand_bounds(self, bounds, means):
        new_bounds = []
        for i in range(len(means)):
            list = []
            for j in range(len(means[i])):
                low_lim = means[i][j] - (bounds[i][j] - means[i][j])
                if low_lim < 0:
                    low_lim = 0
                list.append(sample([low_lim, means[i][j], bounds[i][j]], 1)[0])
            new_bounds.append(list)
        return new_bounds

    def get_means(self):
        means = []
        for i in range(len(self.counts)):
            curr = []
            for j in range(len(self.counts[i])):
                curr.append(self.rewards[i][j] / self.counts[i][j])
            means.append(curr)
        return means

    def get_kl(self, i, j, means, max_cluster):
        sum = 0
        for k in range(len(self.counts[i])):
            index = max((max(means[max_cluster]) - self.overlap), means[i][k])
            sum += self.counts[i][k] * fun.kl(means[i][k], index)
        sum -= self.counts[i][j] * fun.kl(means[i][j], max((max(means[max_cluster]) - self.overlap), means[i][j]))
        return sum

    def get_limit(self, index, max_mean_index, q, means):
        kl_sum = self.get_kl(index[0], index[1], means, max_mean_index[0])
        max_q_prod = self.counts[index[0]][index[1]] * fun.kl(means[index[0]][index[1]], q)
        result = kl_sum + max_q_prod
        return result

    def get_max_index(self, l, isq, optimal_cluster):
        new_list = self.reshape(l)
        max = np.max(new_list)
        if isq:
            for i in range(len(l)):
                for j in range(len(l[i])):
                    if l[i][j] == max and i == optimal_cluster:
                        return i, j
        max_index = np.argmax(new_list)
        return self.cluster_belongings[max_index], self.arm_belongings[max_index]

    def reshape(self, list):
        new_list = np.zeros(len(self.arm_belongings))
        index = 0
        for i in range(len(list)):
            for j in range(len(list[i])):
                new_list[index] = list[i][j]
                index += 1
        return new_list

    def update(self):
        reward = self.bandit.draw(self.cur_c, self.cur_a)
        regret = self.bandit.regret(self.cur_c, self.cur_a)
        self.counts[self.cur_c][self.cur_a] += 1
        self.rewards[self.cur_c][self.cur_a] += reward
        self.regret.append(regret)
        self.t += 1
