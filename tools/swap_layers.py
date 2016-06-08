
import sys
import cPickle
import lasagne.layers

LAYER_TO_SWAP = 1
RESULT_FILE_NAME = "swapped-BASEPong-TARGET-breakout-3-cl1"

# Specify path to the two networks
network_file_from = open(sys.argv[1], 'r')
network_file_to = open(sys.argv[2], 'r')

net_from = cPickle.load(network_file_from)
net_to = cPickle.load(network_file_to)

print "Loaded network 1: ", net_from
print "Loaded network 2: ", net_to

# Get all layers in both networks
layers_net_from = lasagne.layers.get_all_layers(net_from.l_out)
layers_net_to = lasagne.layers.get_all_layers(net_to.l_out)

# First conv layer
net_layer_from = layers_net_from[LAYER_TO_SWAP].W.get_value()
# net_layer_to = layers_net_to[1].W.get_value()

# print net_layer_from == net_layer_to
print "Swapping layer", LAYER_TO_SWAP

layers_net_to[LAYER_TO_SWAP].W.set_value(net_layer_from)


# Test if the operation is working
# fn = lasagne.layers.get_all_layers(net_from.l_out)
# tn = lasagne.layers.get_all_layers(net_to.l_out)
#
# lf = fn[1].W.get_value()
# lt = tn[1].W.get_value()
#
# print lf is lt
# print lf == lt

result_net = net_to

print "Writing swapped network to file..."
print "Object to write: ", result_net

sys.setrecursionlimit(10000)

result_net_file = open(RESULT_FILE_NAME + '.pkl', 'w')
print "File opened"
cPickle.dump(result_net, result_net_file, -1)
print "Pickle dumped"
result_net_file.close()
print "Done"
