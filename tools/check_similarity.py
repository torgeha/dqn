import sys
import cPickle
import lasagne.layers
"""
Load two networks and see if they are the same. Used when layers is swapped.
"""

# Specify path to the two networks
network_file_from = open(sys.argv[1], 'r')
network_file_to = open(sys.argv[2], 'r')

net_from = cPickle.load(network_file_from)
net_to = cPickle.load(network_file_to)
print "----------------------"
print "Loaded network 1: ", net_from
print "Loaded network 2: ", net_to
print "----------------------"
print "n1 == n2: ", net_from == net_to
print "n1 is n2: ", net_from is net_to
print "----------------------"

# Get all layers in both networks
layers_net_from = lasagne.layers.get_all_layers(net_from.l_out)
layers_net_to = lasagne.layers.get_all_layers(net_to.l_out)

from_1 = layers_net_from[1].W.get_value()
to_1 = layers_net_to[1].W.get_value()

print "Comapring first conv layer"
if (from_1 == to_1).all():
    print "n1 layer == n2 layer: True"
else:
    print "n1 layer == n2 layer: False"
print "----------------------"

from_12 = layers_net_from[2].W.get_value()
to_12 = layers_net_to[2].W.get_value()

print "Comapring second conv layer"
if (from_12 == to_12).all():
    print "n1 layer == n2 layer: True"
else:
    print "n1 layer == n2 layer: False"
print "----------------------"

from_13 = layers_net_from[3].W.get_value()
to_13 = layers_net_to[3].W.get_value()

print "Comapring third conv layer"
if (from_13 == to_13).all():
    print "n1 layer == n2 layer: True"
else:
    print "n1 layer == n2 layer: False"
print "----------------------"

from_14 = layers_net_from[4].W.get_value()
to_14 = layers_net_to[4].W.get_value()

print "Comapring fourth dense (512 units) layer"
print "Length: ", len(from_14)
if (from_14 == to_14).all():
    print "n1 layer == n2 layer: True"
else:
    print "n1 layer == n2 layer: False"
print "----------------------"

from_15 = layers_net_from[5].W.get_value()
to_15 = layers_net_to[5].W.get_value()

print "Comapring fifth (out) layer"
print "Length: ", len(from_15)
if (from_15 == to_15).all():
    print "n1 layer == n2 layer: True"
else:
    print "n1 layer == n2 layer: False"
print "----------------------"