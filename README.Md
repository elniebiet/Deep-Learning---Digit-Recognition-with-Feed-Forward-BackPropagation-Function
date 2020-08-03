FEED-FORWARD BACKPROPAGATION FUNCTION DESCRIPTION

The back propagation function, backPropagationFunction() takes as input parameters (listed above), the actual input and the desired outputs for the input and output layers respectively, of both the training and validation datasets; it also takes the number of neurons for the hidden layer, the number of neurons for the output layer, the learning rate, the MSE tolerance, the momentum factor of the network and the maximum number of epochs allowed per iteration. The function applies Cross-Validation technique to the input, running a particular dataset for more than one pair of input and validation data. Thereby, making it more efficient.  

# How it Works (BackProp Process of the function): 
Actual input/training data is first converted to an N * 1 matrix, depending on the number of features  (e.g 4 for the iris dataset, 784 for the digit classifier), this input is fed into the input layer. Initial weights and biases for the input-hidden and hidden-output layers are initialised. Biases are used in both the hidden and output layers. The delta terms for the weights and biases are also initialised. 
In summary, for each sample, it performs forward propagation, backpropagation after determing the gradients of the weights and biases, updates the weights and biases depending on the specified learning rate (the function determines the learning rate by comparing the current error to the last) and finally returns the updated weights and bias for the validation set, including the minimum MSE value achieved. (Please not that all comments for every step are clearly given within the function).

# INPUT PARAMETERS
The Backpropagation function's input parameters are as follows:
## inputA: 
Actual inputs for the input layer (for training)
## desiredOutD: 
Desired outputs at the output layer for each of the inputs (for training) 
## validA: 
Actual inputs for the input layer (for validation)
## validD: 
Desired outputs at the output layer for each of the inputs (for validation) 
## dataSetLink:  
Link to complete data set for cross validation while training (Optional)
## hidden_Neurons: 
number of neurons in the hidden layer
## outpt_Neurns: 
number of neurons in the output layer
## eta: 
the learning rate for modifying the delta terms weights and biases
## MSETolerance: 
the minimum MSE required.
## alpha: 
the momentum factor to accelerate gradient weights for next epoch
## maxNumEpochs: 
maximum number of epochs
# RETURN VALUES
## Wih: 
Input-Hidden layer weights for validation
## Bh: 
Hidden layer Bias for validation
## Whj: 
Hidden-Output weights for validation
## Bo: 
Output layer bias for validation
## achievedMSE: 
The achieved MSE after training
## numEpochsDone: 
The total number of epochs required to complete a particular training. To get accurate return of this value, the maxNumEpochs parameter must be set to a large unreachable value.


