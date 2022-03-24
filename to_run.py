import klucb
import matplotlib.pyplot as plt

T = 10000
runs = 10

data = [[0.5, 0.7, 0.55], [0.5, 0.3, 0.45]]

bandit = fun.BernoulliBandit(data)

data_rep = []
for i in range(len(data)):
    data_rep.append([0] * len(data[i]))

ord_klucb = klucb.KLUCB(data_rep, False, 0)
flat_klucb = klucb.KLUCB(data_rep, True, 0)

results_ord_klucb = fun.run_experiment(ord_klucb, data_rep, runs, T, bandit, 0)
results_flat_klucb = fun.run_experiment(flat_klucb, data_rep, runs, T, bandit, 1)

plt.figure(figsize=(12, 8))

plt.plot(results_ord_klucb[0], label='KL-UCB', color='tab:green')
plt.errorbar([i for i in results_ord_klucb[1]], [results_ord_klucb[0][i] for i in results_ord_klucb[1]],
             yerr=results_ord_klucb[2], fmt='o', capsize=6, color='tab:green')

plt.plot(results_flat_klucb[0], label='Flat KL-UCB', color='tab:orange')
plt.errorbar([i for i in results_flat_klucb[1]], [results_flat_klucb[0][i - 100] for i in results_flat_klucb[1]],
             yerr=results_flat_klucb[2], fmt='o', capsize=6, color='tab:orange')

plt.legend(loc='upper left')
plt.xlabel("Trials")
plt.ylabel("Cumulative Regret")
plt.title("Average Cumulative Regret over "
          + str(runs) + " Runs")
plt.show()
