import numpy as np

# 
class NeuralNetwork(object):
    def __init__(self, epochs=1000, batch_size=None, neural_numbers=[2]):
        self.epochs = epochs
        self.batch_size = batch_size
        self.neural_numbers=neural_numbers
        self.layers=len(self.neural_numbers)+1
        np.random.seed(77)

    def fit(self,X,y):
        self.X,self.y = X,y
        self.initial_weight()
    
    def forward(self,X):

        output_list = []
        input_x = X

        for layer in range(self.layers):

            #first hidden layer 
            cur_weight = self.weight_list[layer]
            cur_bias = self.bias_list[layer]

            # Calculate the output for current layer
            if layer == 0:
                output = self.neuron_output(cur_weight,input_x,cur_bias)
            elif layer == 1:
                output = self.neuron_output_second(cur_weight,input_x,cur_bias)    

            # The current output will be the input for the next layer.
            input_x =  output
            output_list.append(output)

        return output_list

    def predict(self,X):
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

    # Classical sigmoid activation functions are used in every layer in this network
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    # Calculate the output for this layer
    def neuron_output(self,w,x,b):
        wx=np.dot(x, w)
        return self.sigmoid( wx + b)

    # Calculate the output for second hidden layer
    def neuron_output(self,w,x,b):
        wx=np.dot(x, w)
        return self.softmax( wx + b)    

    def der_last_layer(self,loss_last,output,input_x):
        # softmax for last layer
        softmax_x = softmax(input_x)
        sigmoid_der=self.sigmoid_der(output)
        loss = sigmoid_der*loss_last
        dW = np.dot(input_x.T, loss)
        db = np.sum(loss, axis=0, keepdims=True)
        return loss,dW,db

    
    '''
    def der_hidden_layer(self,loss_last,output,input_x,weight):
        loss = self.sigmoid_der(output) * np.dot(loss_last,weight.T)
        db = np.sum(loss, axis=0, keepdims=True)
        dW = np.dot(input_x.T, loss)
        return loss,dW,db
    '''

    def softmax(self,y):
        #return np.argmax(y,axis=1)
        return np.exp(y) / np.sum(np.exp(y), axis=0)

    # Loss Function 
    def cross_entropy(X,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        p = softmax(X)
        # We use multidimensional array indexing to extract 
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss

    