from math import exp
from random import random
 
###############################################################################
# Initialize the network
###############################################################################
def rand_gen():
    #using DR. AJ's random generator function
    return 1-2*random()

def create_layer(n_in_nodes,n_out_nodes,bias=True):
    #utility that creates a layer
    if bias:
        n_in_nodes += 1
    layer = [{'weights':[rand_gen() for i in range(n_in_nodes)]} for i in range(n_out_nodes)]
    return layer

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    network.append(create_layer(n_inputs,n_hidden))
    network.append(create_layer(n_hidden,n_outputs))
    return network

###############################################################################
# Forward propogate inputs
###############################################################################

# neuron activation
def activate(weights, inputs):
    #the last weight serves as a bias
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# neuron Transfer
def squash(alpha,activation):
    #sigmoid function
    return 1.0/(1.0 + exp(-alpha * activation))

#forward propogation
def forward_propagate(network, row, alpha=1.0):
    #loop through the network
    inputs = row
    for layer in network:
        new_inputs = []
        #apply the activation and transfer functions
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = squash(alpha,activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
 
###############################################################################
# Back propogate error
###############################################################################

def transfer_derivative(output):
    return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    # loop backwards through the layers
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            # case not output to hidden
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            #case from output to hidden
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            # assign credit for error
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
###############################################################################
# Train Network
###############################################################################
 
# Update network weights with error
def update_weights(network, inputs, nu):
    # inputs are either the raw data inputs or the weighted neuron outputs
    for i in range(len(network)):
        #inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += nu * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += nu * neuron['delta']   
              
# Train a network for a fixed number of epochs
def train_network(network, data, nu, max_epoch,n_outputs,error_threshold=.5):
    """Data assumes that a structure of [design_matrix,target_matrix]"""
    for epoch in range(max_epoch):
        sum_error = 0
        for row in data:
            inputs = row[0]
            outputs = forward_propagate(network, inputs)
            targets = row[1]
            sum_error += sum([(targets[i]-outputs[i])**2 for i in range(len(targets))])
            backward_propagate_error(network, targets)
            update_weights(network, inputs, nu)
        if sum_error <= error_threshold:
            print "Achieved error threshold %.3f after %d epochs" %(error_threshold,epoch)
            break
        if epoch%100==0:
            print '>epoch=%d, l_rate=%.2f error=%.3f' % (epoch, nu, sum_error)
    else:
        print "Reached maximum number of epochs (%d). Error: =%.3f"%(max_epoch,sum_error)
 
###############################################################################
# Predict/Util
###############################################################################

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
	
	
def inspect_weights(network):
    for i,layer in enumerate(network):
        # gets to the layers of the network
        if i == 0:
            l = 'input'
            n_l = 'hidden'+str(i)
        elif i == len(network)-1:
            l='hidden'+str(i-1)
            n_l = 'output'
        else:
            l = 'hidden'+str(i-1)
            n_l = 'hidden'+str(i)
        for g,neuron in enumerate(layer):
            #g = neuron number in the layer
            for j,weight in enumerate(neuron['weights']):
                if j == len(neuron['weights'])-1:
                    #this is the bias
                    print "Bias for %s_%d = %.4f"%(l,g,weight)
                else:
                    #these are the connection weights
                    print "Weight from %s_%d to %s_%d = %.4f"%(l,g,n_l,j,weight)               
        

###############################################################################
# Main
###############################################################################

def main(n_inputs=None,n_hidden=None,n_outputs=None,alpha=1.0,nu=.5):

        
    training_set = [
                    [[0,0],[1,0]],
                    [[1,0],[0,1]],
                    [[0,1],[0,1]],
                    [[1,1],[1,0]]]
    
    # train the model
    if n_inputs==None:
        n_inputs = len(training_set[0][0])
    if n_outputs==None:
        n_outputs = len(training_set[0][1])
        
    nn = initialize_network(n_inputs,n_hidden,n_outputs)
    train_network(nn,training_set,nu,5000,n_outputs,error_threshold=.1)
    	
    for row in training_set:
   	prediction = predict(nn, row[0])
   	print 'Expected=',row[1].index(max(row[1])),' Got=', prediction
    
    inspect_weights(nn)
if __name__ == "__main__":
    
    main(n_hidden=2,nu=.1)


