"""
Calculates the difference score between a layer in two networks

Usage:

layer_difference.py LAYER PICKLED_NN_FILE1 PICKLED_NN_FILE2

se
http://lasagne.readthedocs.org/en/latest/modules/layers/cuda_convnet.html
for hvordan weights er hentet ut fra lasagne layers
"""

import sys
import cPickle
import lasagne.layers
import numpy as np


def normalize(n, max, min):
    return ((n - min) / (max - min))


# Load network
layer_to_check = int(sys.argv[1])
net_file1 = open(sys.argv[2], 'r')
net_file2 = open(sys.argv[3], 'r')

network1 = cPickle.load(net_file1)
network2 = cPickle.load(net_file2)

print network1
print network2

# Get all layers from output layer and backwards
layers_net1 = lasagne.layers.get_all_layers(network1.l_out)
layers_net2 = lasagne.layers.get_all_layers(network2.l_out)

# Specified conv layer
weights_net1 = layers_net1[layer_to_check].W.get_value()
weights_net2 = layers_net2[layer_to_check].W.get_value()

# count = 1

# Assume similar structure for all netowkrs
print "Filters net 1: ", len(range(weights_net1[0].shape[0]))
print "Filters net 2: ", len(range(weights_net2[0].shape[0]))
print "Channels/timesteps net 1: ", len(range(weights_net1[0].shape[1]))
print "Channels/timesteps net 2: ", len(range(weights_net2[0].shape[1]))

kernel_diffs = []
diff_indices = {}

net1_avgs = []
net2_avgs = []

pong1_unadjusted_filters = [2, 5, 6, 8, 10, 22, 24, 30]
pong2_unadjusted_filters = [4, 5, 16, 17, 18, 25, 29]

# Assuming same shape and size for both layers
for c in range(weights_net1.shape[1]): # channels/time-steps

    for f in range(weights_net1.shape[0]): # filters

        img_net1 = weights_net1[f, c, :, :]
        # img_net2 = weights_net2[f, c, :, :]


        # Extract filter if unadjusted
        if f in pong2_unadjusted_filters:
            continue

        # Normalize data
        norm_img1 = []
        # norm_img2 = []
        min1 = img_net1.min()
        # min2 = img_net2.min()
        max1 = img_net1.max()
        # max2 = img_net2.max()

        # Normalize img from first layer
        for i in range(len(img_net1)):
            for j in range(len(img_net1[0])):

                n1 = normalize(img_net1[i][j], max1, min1)
                # n2 = ((img_net2[i][j] - min2) / (max2 - min2))
                norm_img1.append(n1)
                # norm_img2.append(n2)
                # print img_net1[i][j], " --> ", n1

        # contains the diff with current first img and all other filters in second layer
        # This contains the diff for 32x4 filters of second layer
        diff_avg_current_first_img = []

        # Compare with all filters in other layer
        for c2 in range(weights_net2.shape[1]): # channels/time-steps

            for f2 in range(weights_net2.shape[0]): # filters

                # image from second layer
                img_net2 = weights_net2[f2, c2, :, :]

                # Extract unadjusted filters
                # if f2 in pong2_unadjusted_filters:
                #     continue

                norm_img2 = []
                min2 = img_net2.min()
                max2 = img_net2.max()

                # Find pixelwise difference between normalized img arrays
                # and append to diff array
                pixelwise_diff = []

                # Normalize img from second layer
                for i in range(len(img_net2)):
                    for j in range(len(img_net2[0])):

                        n2 = normalize(img_net2[i][j], max2, min2)
                        norm_img2.append(n2)

                # pixelwise diff between current two images
                img_diff = []
                for i in range(len(norm_img1)): # assume same length
                    d = norm_img1[i] - norm_img2[i]
                    a = np.absolute(d)
                    img_diff.append(a)

                # Average diff of current imgs
                diff_avg_current_first_img.append(np.average(img_diff))

        # One image from first layer is done, find the lowest diff and append
        min_index = np.argmin(diff_avg_current_first_img)

        # This counts the filter occurrence
        diff_indices[str(min_index)] = diff_indices.get(str(min_index), 0) + 1
        kernel_diffs.append(diff_avg_current_first_img[min_index])

# print diff_indices
sum = 0
for key, value in diff_indices.iteritems():
    sum += value
    print key, ":", value

print "Filters used", len(diff_indices)
print sum
# print kernel_diffs
layer_diff = np.average(kernel_diffs)
print "Difference: ", layer_diff