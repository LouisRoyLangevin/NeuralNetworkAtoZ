# NeuralNetworkAtoZ
This is a python script that allows the user to create neural networks of any desired sizes.  Back-propagation and gradient descent algorithms have been implement from A to Z without using any library.


#How to use it
1) Download the main.py file on your own device.  main.py is located in the main branch of this repository.
2) Go to the bottom of the file and write your own piece of code in the "User's part".
3) To know what to write, read the rest of this Readme file.


Create a new neural network
You can create a neural network writing

"NN = CreateNN(arr)"

where arr = [a0,a1,...,an] and ak's are natural numbers.  This will create a neural network with n+1 layers (n-1 hidden layers) with respective sizes a0,a1,...,an, where a0 is the input layer's size and an in the output layer's size.

The weights and biases will be Unif[0,1] generated (uniformly between 0 and 1).


You also have the option of setting the initial weights and biases by yourself.  You can simply write

"NN = NeuralNetwork(arr,weights,biases)"

where arr = [a0,a1,...,an] as above, weights and biases are multidimensional arrays that contain the initial weights and biases of NN.  The way weights are biases are organized is explained later in this Readme file.



##Training your neural network

First, you want to add (xi,yi) vectors to the training set of your neural network NN.  To do so, you can use the method NN.addToSet(xi,yi) where xi,yi are arrays of sizes a0,an respectively.  You have to use this method once for each tuple (xi,yi) that you want to train your neural network with.


Once your training set is complete, you can start the gradient descent algorithm with the NN.train() method.  This will adjust the weights and biases of your neural network NN and minimize its loss value NN.loss().



##How weights and biases are organized

Given a neural network NN with n+1 layers with sizes a0,...,an respectively (in order from input to output), we will denote the i'th neuron of the m'th layer as A_mi.  Then the weight of the edge going from A_mi to A_(m+1)j is stored in weights[m][j][i].  Notice that the index of j goes before the index of i.  This is normal, it facilitates the back-propagation algorithm.  

On the other hand, the bias of neuron A_mi is stored in biases[m-1][i].  The reason for m-1 instead of m is that the input neurons don't have biases, so start our biases on the second layer (so, usually, the first hidden layer).
