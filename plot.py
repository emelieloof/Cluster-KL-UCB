import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import numpy as np

data1 = [[0.65, 0.7, 0.8, 0.6, 0.75], [0.6, 0.65, 0.75, 0.55, 0.7], [0.25, 0.3, 0.4, 0.2, 0.35],
       [0.25, 0.3, 0.4, 0.2, 0.35], [0.25, 0.3, 0.4, 0.2, 0.35]] # 75%
data2 = [[0.65, 0.7, 0.8, 0.6, 0.75], [0.5, 0.55, 0.65, 0.45, 0.6], [0.25, 0.3, 0.4, 0.2, 0.35],
       [0.25, 0.3, 0.4, 0.2, 0.35], [0.25, 0.3, 0.4, 0.2, 0.35]] #25%
data3 = [[0.65, 0.7, 0.8, 0.6, 0.75], [0.4, 0.45, 0.55, 0.35, 0.5], [0.25, 0.3, 0.4, 0.2, 0.35],
        [0.25, 0.3, 0.4, 0.2, 0.35], [0.25, 0.3, 0.4, 0.2, 0.35]] # -25%

data = [data1, data2, data3]

mpl.style.use('ggplot')

runs = 50
i = 0 # the data consist of results for all three expirimets, i = 0 results for 75% overlap, i = 1 25% overlap, i = 2 -25% overlap


result_KL = json.load(open('cVSkl_KL.json'))
cr_KL = result_KL['KLUCB'][i]['result'][0]
er_KL = result_KL['KLUCB'][i]['result'][1]
x_KL = result_KL['KLUCB'][i]['result'][2]

result_cKL = json.load(open('cVSkl_C.json'))
cr_ckl = result_cKL['Clustered KL-UCB'][i]['result'][0]
er_cKL = result_cKL['Clustered KL-UCB'][i]['result'][1]
x_cKL = result_cKL['Clustered KL-UCB'][i]['result'][2]


fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(cr_ckl, label='Cluster KL-UCB', color='tab:orange')
ax1.errorbar([i for i in x_cKL], [cr_ckl[i] for i in x_cKL], yerr=er_cKL, fmt='o', capsize=6,
             color='tab:orange')

ax1.plot(cr_KL, label='KL-UCB', color='tab:green')
ax1.errorbar([i for i in x_KL], [cr_KL[i] for i in x_KL], yerr=er_KL, fmt='o', capsize=6,
             color='tab:green')

l, b, h, w = .71, .12, .18, .18
ax2 = fig.add_axes([l, b, w, h])
colors = ['tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:brown']
x_d = [np.arange(0,5), np.arange(5, 10), np.arange(10, 15), np.arange(15, 20), np.arange(20, 25)]

for j in range(5):
    ax2.plot(x_d[j], np.sort(data[i][j]), 'o', color=colors[j], ms=2)
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)
ax1.legend(loc='upper left')
ax2.set_ylim(0,1)
ax1.set_ylim(-25, 350)
ax1.set_xlabel("Trials", fontsize=13)
ax1.set_ylabel("Cumulative Regret", fontsize=13)
ax1.set_title("Average Cumulative Regret over "
              + str(runs) + " Runs", fontsize=15)

plt.savefig('cVSKL_neg25.pdf')
plt.show()
