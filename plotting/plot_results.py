"""
Usage:

plot_results.py <nof_results_to_avg_over> <res1_base> <res2_base> ... <res1_target> <res2_target> ...

"""

import numpy as np
import matplotlib.pyplot as plt
import sys


MAX_EPOCHS = 100

nof_results_to_avg_over = int(sys.argv[1])

epochs = None

# Get base results, cut length  and append to list
base_list = []
for i in range(2, nof_results_to_avg_over + 2):
    base = np.loadtxt(open(sys.argv[i], "rb"), delimiter=",", skiprows=1)

    # Get epochs list, same for all. I know its stupid to assign every time
    epochs = base[:, 0][:MAX_EPOCHS]

    # Create list with [avg_score_list, q_values_list] all clipped to 100 elements
    base_res = [base[:, 3][:MAX_EPOCHS], base[:, 4][:MAX_EPOCHS]]
    base_list.append(base_res)

# Get target results, cut length and append to list
target_list = []
for i in range(nof_results_to_avg_over + 2, (nof_results_to_avg_over * 2) + 2):
    target = np.loadtxt(open(sys.argv[i], "rb"), delimiter=",", skiprows=1)

    target_res = [target[:, 3][:MAX_EPOCHS], target[:, 4][:MAX_EPOCHS]]
    target_list.append(target_res)


print len(base_list)
print len(base_list[0])
print len(base_list[0][0])
print len(base_list[0][1])


base_score_result = []
base_q_result = []
target_score_result = []
target_q_result = []


# Avg the base results
for i in range(0, MAX_EPOCHS): # TODO: check if 99 or 100 epochs

    current_avg_base_res = []
    current_q_base_vals = []

    current_avg_target_res = []
    current_q_target_vals = []

    # each result to avg
    for j in range(0, nof_results_to_avg_over):

        # Find base values and append
        current_avg_base_res.append(base_list[j][0][i]) # append score
        current_q_base_vals.append(base_list[j][1][i])

        # Find target values and append
        current_avg_target_res.append(target_list[j][0][i])
        current_q_target_vals.append(target_list[j][1][i])

    # print current_avg_base_res
    # print current_q_base_vals

    # Append the average to results lists
    base_score_result.append(np.average(current_avg_base_res))
    base_q_result.append(np.average(current_q_base_vals))
    target_score_result.append(np.average(current_avg_target_res))
    target_q_result.append(np.average(current_q_target_vals))

    # print base_score_result
    # print base_q_result


# results = np.loadtxt(open(sys.argv[1], "rb"), delimiter=",", skiprows=1)

# plt.subplot(1, 2, 1)

# Get data
# epochs = results[:, 0][:MAX_EPOCHS]
# avg_score_per_episode = results[:, 3][:MAX_EPOCHS]

# Plot average score
# plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-')
plt.plot(epochs, base_score_result, 'r-', linewidth=2, label="Base")
plt.plot(epochs, target_score_result, 'b-', linewidth=2, label="Target")
plt.legend(loc=4)
#plt.title("Average score per episode")
plt.grid()

plt.xlabel('Epochs')
# plt.ylabel('Average score per episode')
plt.ylabel('Average score per episode')
#plt.ylim([0, 250])

plt.tight_layout()
#plt.savefig('pong_test.png')

#sys.exit(0)

plt.show()

# plt.subplot(1, 2, 2)

# Get data
# q_values = results[:, 4][:MAX_EPOCHS]

# Plot q_values
# plt.plot(results[:, 0], results[:, 4], '-')
plt.plot(epochs, base_q_result, 'r-', linewidth=2, label="Base")
plt.plot(epochs, target_q_result, 'b-', linewidth=2, label="Target")
plt.legend(loc=4)
#plt.title("Average action value")
plt.grid()

plt.xlabel('Epochs')
plt.ylabel('Average action value per episode')
#plt.ylim([0, 4])
plt.tight_layout()

plt.show()
