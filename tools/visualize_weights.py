""" Utility to plot the first layer of convolutions learned by
the Deep q-network.

(Assumes dnn convolutions)

Usage:

plot_filters.py PICKLED_NN_FILE

se
http://lasagne.readthedocs.org/en/latest/modules/layers/cuda_convnet.html
for hvordan weights er hentet ut fra lasagne layers
"""

import sys
import matplotlib.pyplot as plt
import cPickle
import lasagne.layers

# Load network
net_file = open(sys.argv[1], 'r')
network = cPickle.load(net_file)

print network

# Get all layers from output layer and backwards
layers = lasagne.layers.get_all_layers(network.l_out)

# First conv layer
weights = layers[1].W.get_value()

count = 1

# Assume similar structure for all netowkrs
print "Filters: ", len(range(weights[0].shape[0]))
print "Channels/timesteps: ", len(range(weights[0].shape[1]))

plt.subplots_adjust(left=0.06,
                    bottom=0.75,
                    right=0.94,
                    top=0.97,
                    wspace=0.20,
                    hspace=0.21)

for c in range(weights.shape[1]): # channels/time-steps

    for f in range(weights.shape[0]): # filters

        plt.subplot(weights.shape[1], weights.shape[0], count)

        # plt.xlabel("Some X label")

        # plt.title("T" + str(c) + "-" + "w" + str(f))
        plt.title(f)

        img = weights[f, c, :, :]

        # intImg = img.astype('uint8')

        if (c == 0 and f == 0):
            print img
            print type(img)
            import numpy as np
            print np.dtype(img[0][0])

            print (np.amin(img))
            print(np.amax(img))

        # TODO: Resize??

        plt.imshow(img, vmin=img.min(), vmax=img.max(),
                   interpolation='none', cmap='gray')

        # Dont display x/y ticks
        plt.xticks(())
        plt.yticks(())

        count += 1
        # print count
plt.show()
