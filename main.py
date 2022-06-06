#Example to run experiment (this example comes from section 3)

import bandit as b
import algorithms as algo
import functions as fun
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


instance1 = [[0.65, 0.7, 0.8, 0.6, 0.75], [0.35, 0.4, 0.5, 0.3, 0.45], [0.25, 0.3, 0.4, 0.2, 0.35],
         [0.25, 0.3, 0.4, 0.2, 0.35], [0.25, 0.3, 0.4, 0.2, 0.35]]

instance2 = [[0.65, 0.7, 0.8, 0.6, 0.75], [0.55, 0.6, 0.7, 0.5, 0.65], [0.25, 0.3, 0.4, 0.2, 0.35],
         [0.25, 0.3, 0.4, 0.2, 0.35], [0.25, 0.3, 0.4, 0.2, 0.35]]

d = [instance1, instance2]
T = 10000
runs = 50



results = []

for data in d:
    bandit = b.BernoulliBandit(data)
    data_rep = []
    for i in range(len(data)):
        data_rep.append([0] * len(data[i]))

    max_kl = algo.TwolevelKLUCB(data_rep, 'max', bandit)
    result_max = fun.run_experiment(max_kl, data_rep, runs, T, 0)

    min_kl = algo.TwolevelKLUCB(data_rep, 'min', bandit)
    result_min = fun.run_experiment(min_kl, data_rep, runs, T, 1)

    avg_kl = algo.TwolevelKLUCB(data_rep, 'avg', bandit)
    result_avg = fun.run_experiment(avg_kl, data_rep, runs, T, 2)

    kl = algo.KLUCB(data, bandit)
    result_kl = fun.run_experiment(kl, data_rep, runs, T, 3)

    twol_ts = algo.TwolevelTS(data, bandit)
    result_twolts = fun.run_experiment(twol_ts, data_rep, runs, T, 4)

    ts = algo.TS(data_rep, bandit)
    result_ts = fun.run_experiment(ts, data_rep, runs, T, 5)
    results.append([result_max, result_min, result_avg, result_kl, result_twolts, result_ts])


mpl.style.use('ggplot')
colors = ['tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:red']
labels = ['2-level KL-UCB max', '2-level KL-UCB min', '2-level KL-UCB avg', 'KL-UCB', '2-level TS', 'TS']



fig, ax1 = plt.subplots(figsize=(12, 8))
result = results[0]

for i in range(len(result)):
    ax1.plot(result[i][0], label=labels[i], color=colors[i])
    ax1.errorbar([j for j in result[i][2]], [result[i][0][j] for j in result[i][2]], yerr=result[i][1], fmt='o', capsize=6,
                 color=colors[i])

l, b, h, w = .13, .55, .15, .15
ax2 = fig.add_axes([l, b, w, h])

x_d = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15), np.arange(15, 20), np.arange(20, 25)]
for i in range(len(instance1)):
    ax2.plot(x_d[i], np.sort(instance1[i]), 'o', color=colors[i], ms=2)
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)
ax1.legend(loc='upper left', fontsize=15)
ax2.set_ylim(0,1)
ax1.set_xlabel("Trials", size=20)
ax1.set_ylabel("Cumulative Regret", size=20)

plt.show()


fig, ax1 = plt.subplots(figsize=(12, 8))
result = results[1]

for i in range(len(result)):
    ax1.plot(result[i][0], label=labels[i], color=colors[i])
    ax1.errorbar([j for j in result[i][2]], [result[i][0][j] for j in result[i][2]], yerr=result[i][1], fmt='o', capsize=6,
                 color=colors[i])

l, b, h, w = .13, .55, .15, .15
ax2 = fig.add_axes([l, b, w, h])

x_d = [np.arange(0,5), np.arange(5, 10), np.arange(10, 15), np.arange(15, 20), np.arange(20, 25)]
for i in range(len(instance2)):
    ax2.plot(x_d[i], np.sort(instance2[i]), 'o', color=colors[i], ms=2)
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)
ax1.legend(loc='upper left', fontsize=15)
ax2.set_ylim(0,1)
ax1.set_xlabel("Trials", size=20)
ax1.set_ylabel("Cumulative Regret", size=20)

plt.show()
