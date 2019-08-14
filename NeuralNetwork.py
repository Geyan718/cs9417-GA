import numpy as np

#class NeuralNetwork():

    #neural_numbers = [4]
    #input_dim = 4
    #output_dim = 3
    #layers = len(NeuralNetwork.neural_numbers) + 1
    
def forward(X, weight_list, biases_list):

    input_x = X

    for layer in range(2):

        #first hidden layer 
        cur_weight = weight_list[layer]
        cur_bias = biases_list[layer]

        # Calculate the output for current layer
        if layer == 0:
            output = neuron_output(cur_weight,input_x,cur_bias)
        elif layer == 1:
            output = neuron_output_second(cur_weight,input_x,cur_bias)    

        # The current output will be the input for the next layer.
        input_x = output

    return output

# Classical sigmoid activation functions are used in every layer in this network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# Calculate the output for this layer
def neuron_output(w,x,b):
    wx=np.dot(x, w)
    return sigmoid( (wx + b).astype(float))

# Calculate the output for second hidden layer
def neuron_output_second(w,x,b):
    wx=np.dot(x, w)
    return softmax( (wx + b).astype(float))    

def softmax(y):
    #return np.argmax(y,axis=1)
    return np.exp(y) / np.sum(np.exp(y), axis=0)

# Loss Function 
def cross_entropy(predictions,labels):
    epsilon = 1e-12
    n = labels.shape[0]
    clipped_preds = np.clip(predictions, epsilon, 1. - epsilon)
    loss = -np.sum(labels*np.log(clipped_preds))/n
    return loss

    