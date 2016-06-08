"""
Plot learning loss. 6k-10k datapoints is too much, using a moving avergae of datapoints to downsample

plot_results.py <nof_results_to_avg_over> <learning1_base> <learning2_base> ... <learning1_target> <learning2_target> ...
"""


import numpy as np
import matplotlib.pyplot as plt
import sys

AVG_FILTER = 50 # For breakout
# AVG_FILTER = 10 # For pong

# def moving_average(a, n=AVG_FILTER) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

def moving_average(a, n=AVG_FILTER):

    avg_res = []

    for i in range(n, len(a), n):
        avg_res.append(np.average(a[i-n:i]))

    return avg_res
# print moving_average(range(10), 2)

nof_results_to_avg_over = int(sys.argv[1])

min_length = sys.maxint

base_list = []
for i in range(2, nof_results_to_avg_over + 2):
    base = np.loadtxt(open(sys.argv[i], "rb"), delimiter=",", skiprows=1)

    bl = base[:, 0] #[:MAX_EPOCHS]

    # Used to clip all listst to the shortest
    if len(bl) < min_length:
        min_length = len(bl)

    base_list.append(bl)

target_list = []
for i in range(nof_results_to_avg_over + 2, (nof_results_to_avg_over * 2) + 2):
    target = np.loadtxt(open(sys.argv[i], "rb"), delimiter=",", skiprows=1)

    tl = target[:, 0]

    if len(tl) < min_length:
        min_length = len(tl)

    target_list.append(tl)

# Clip length of all lists
for i in range(nof_results_to_avg_over):
    base_list[i] = base_list[i][:min_length]
    target_list[i] = target_list[i][:min_length]

# Avg base and target lists
base_avg = []
target_avg = []
for i in range(min_length):

    current_base = []
    current_target = []

    for j in range(nof_results_to_avg_over):
        current_base.append(base_list[j][i])
        current_target.append(target_list[j][i])


    base_avg.append(np.average(current_base))
    target_avg.append(np.average(current_target))


# Reduce size of base and target lists by moving average.
# Extract first value, for prettier graph
base_first = base_avg[0]
target_first = target_avg[0]

base_moved = moving_average(base_avg)
target_moved = moving_average(target_avg)



# plt.plot(range(len(base_avg)), base_avg, 'r-', linewidth=1, label="Base")
# plt.plot(range(len(target_avg)), target_avg, 'b-', linewidth=1, label="Target")
# plt.legend()
# plt.title("Mean loss moving average per episode")
# plt.grid()
#
# plt.xlabel('Episode')
# # plt.ylabel('Average score per episode')
# plt.ylabel('Mean loss')
# #plt.ylim([0, 250])
#
# plt.show()

x_axis = range(0, len(base_avg), AVG_FILTER)
x_axis = x_axis[:len(base_moved)]

plt.plot(x_axis, base_moved, 'r-', linewidth=1, label="Base")
plt.plot(x_axis, target_moved, 'b-', linewidth=1, label="Target")
plt.legend(loc=4) # legend displayed at bottom right
# plt.legend() # Legend displayed at top right
# plt.title("Mean loss moving average per episode")
plt.grid()

plt.xlabel('Episode')
# plt.ylabel('Average score per episode')
plt.ylabel('Mean loss')
#plt.ylim([0, 250])

plt.tight_layout()

plt.show()