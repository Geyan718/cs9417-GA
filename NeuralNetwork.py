import numpy as np

class NeuralNetwork():

    neural_numbers = [4]
    input_dim = 4
    output_dim = 3
    layers = len(self.neural_numbers) + 1
    
    def forward(cls, X, weight_list, biases_list):

        input_x = X

        for layer in range(cls.layers):

            #first hidden layer 
            cur_weight = self.weight_list[layer]
            cur_bias = self.bias_list[layer]

            # Calculate the output for current layer
            if layer == 0:
                output = self.neuron_output(cur_weight,input_x,cur_bias)
            elif layer == 1:
                output = self.neuron_output_second(cur_weight,input_x,cur_bias)    

            # The current output will be the input for the next layer.
            input_x = output

        return output
    '''
    def predict(cls,X):
        output_list = self.forward(X)
        pred_y = self.softmax(output_list[-1])
        return pred_y

    def initial_weight(self):

        if self.X is not None and self.y is not None:
            x=self.X
            y=self.y
            input_dim = x.shape[1]
            output_dim = y.shape[1]

            number_NN = self.neural_numbers+[output_dim]

            weight_list,bias_list = [],[]
            last_neural_number = input_dim     

            for cur_neural_number in number_NN:
                
                # The dimension of weight matrix is last neural number * current neural number
                weights = np.random.randn(last_neural_number, cur_neural_number)
                # The number of dimension for bias is 1 and the number of current neural
                bias = np.zeros((1, cur_neural_number))

                last_neural_number=cur_neural_number

                weight_list.append(weights)
                bias_list.append(bias)

            self.weight_list=weight_list
            self.bias_list=bias_list
    '''
    # Classical sigmoid activation functions are used in every layer in this network
    def sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))
    
    # Calculate the output for this layer
    def neuron_output(cls,w,x,b):
        wx=np.dot(x, w)
        return cls.sigmoid( wx + b)

    # Calculate the output for second hidden layer
    def neuron_output_second(cls,w,x,b):
        wx=np.dot(x, w)
        return cls.softmax( wx + b)    

    def softmax(cls,y):
        #return np.argmax(y,axis=1)
        return np.exp(y) / np.sum(np.exp(y), axis=0)

    # Loss Function 
    def cross_entropy(cls, predictions,labels):
        epsilon = 1e-12
        n = labels.shape[0]
        clipped_preds = np.clip(predictions, epsilon, 1. - epsilon)
        loss = -np.sum(labels*np.log(clipped_preds))/n
        return loss

    