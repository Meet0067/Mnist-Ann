#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

#Part-1 Data PreProcessing Phase
from keras.datasets.mnist import load_data

(X_train,y_train),(X_test,y_test) = load_data()

X_train = X_train.reshape((X_train.shape[0],784))
X_test = X_test.reshape((X_test.shape[0],784))
y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.175)

X_train = X_train / 255
X_test  = X_test  / 255
X_val   = X_val   / 255


#Part-2 Model Creation Phase

#Neural Network Initialization

def sigmoid(z):
    A =  (1/(1+np.exp(-z)))
    cache = (z)
    
    return A,cache

def tanh_(z):
    A = np.tanh(z)
    cache = (z)
    
    return A,cache

def relu(Z): 
    A = np.maximum(0,Z)  
    cache = Z 
    
    return A, cache

def softmax(Z):    
    expZ = np.exp(Z - np.max(Z))
    A = expZ / (np.sum(expZ,axis=0,keepdims=True))
    cache = Z
    return A,cache

def relu_backward(dA, cache):    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.    
    dZ[Z <= 0] = 0  # When z <= 0, you should set dz to 0 as well. 
 
    return dZ

def sigmoid_backward(dA, cache):      
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def tanh_backward(A,dA):       
    dZ = dA *(1-np.power(A,2))
    
    return dZ

def layers(X,Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return n_x,n_y

def initialize(layer_dims):    
    np.random.seed(2)    
    parameters = { }
    L = len(layer_dims)
    for i in range(1,L):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2/(layer_dims[i-1]))
        parameters["b"+str(i)] = np.zeros((layer_dims[i],1))
    
    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)
    
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation,keep_prob = 1):
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev, W, b)
        A,activation_cache = sigmoid(Z)
        D = np.random.rand(A.shape[0],A.shape[1])                                         
        D =  D < 1        
        A =  A*D  
                                             
    elif activation == "softmax":
        Z,linear_cache = linear_forward(A_prev, W, b)
        A,activation_cache = softmax(Z)   
        D = np.random.rand(A.shape[0],A.shape[1])                                         
        D =  D < 1        
        A =  A*D  
        
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev, W, b)
        A,activation_cache = relu(Z) 
        D = np.random.rand(A.shape[0],A.shape[1])                                         
        D =  D < keep_prob                                         
        A =  A*D                                         
        A = A/keep_prob   
    cache = (linear_cache,activation_cache)
    
    return A,cache,D

def L_model_forward(X,parameters,keep_prob=1):    
    caches = []
    A = X
    L= len(parameters) // 2
    Ds ={}
    for l in range(1,L):
        A_prev = A
        A, cache ,D=linear_activation_forward(A_prev, parameters['W'+str(l)],parameters['b'+str(l)] , activation="relu",keep_prob = keep_prob)
        caches.append(cache)
        Ds["D" + str(l)] = D
        
    AL ,cache,D = linear_activation_forward(A, parameters['W'+str(L)],parameters['b'+str(L)] , activation="softmax",keep_prob = keep_prob)
    caches.append(cache)
    Ds["D"+str(L)] = D
    Ds["D0"] = Ds["D"+str(L)]
    return AL,caches,Ds

def compute_cost(AL,Y):
   
    cost = -np.mean(Y * np.log(AL + 1e-8))  
    
    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd=0.1):
    
    m = Y.shape[1]
    L= len(parameters) // 2
    W1 = parameters["W1"].T
    
    sum_ = np.sum(np.square(W1))
    for l in range(2,L+1):   
      sum_ = sum_ + np.sum(np.square(parameters['W'+str(l)+'']))   

    
    cross_entropy_cost = compute_cost(AL, Y) # This gives you the cross-entropy part of the cost
    
    L2_regularization_cost =  (1/m)*(sum_)
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def linear_backward(dZ,cache,lambd = 0.1):
    A_prev, W, b = cache
    m = A_prev.shape[1]  
    dW = (1/m)*np.dot(dZ,A_prev.T) + (lambd/m)*W
    db = (1/m)*(np.sum(dZ,axis=1,keepdims=True))
    dA_prev = np.dot(W.T,dZ)    
   
    return dA_prev, dW, db

def linear_activation_backward(dA,cache,activation,D,keep_prob = 1):    
    linear_cache, activation_cache = cache    
    if activation == "relu":    
        dA = dA*D
        dA = dA/keep_prob
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
       
    elif activation == "sigmoid": 
        dA = dA*D
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)        
    
    return dA_prev, dW, db
    
def L_model_backward(AL,Y,caches,Ds,keep_prob,lambd = 0.1):
    grads = {}
    L = len(caches)         # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation  
    # dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
   
    # Lth layer (Softmax -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    linear_cache, activation_cache = current_cache
    A_prev, W, b  = linear_cache
    dZ = AL - Y    
    dW = dZ.dot(A_prev.T) / m + (lambd/m)*W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dAPrev = W.T.dot(dZ)
    
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = dAPrev,dW,db
      
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache =  caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,activation ="relu",D = Ds["D" +str(l+1)],keep_prob =keep_prob)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp      

    return grads

def update_params(parameters ,grads,learning_rate): 
    
    L = len(parameters) //2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W"+ str(l+1)] -learning_rate*grads["dW"+ str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+ str(l+1)] -learning_rate*grads["db"+ str(l+1)]
        
    return parameters

def predict(X, y, parameters):    
    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches ,Ds= L_model_forward(X, parameters)
    y_hat = np.argmax(probas.T,axis=1)
    y = np.argmax(y.T,axis=1)
    # convert probas to 0/1 predictions  
    accuracy = (y_hat == y).mean()	 
   
    return accuracy * 100,y_hat

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    
    np.random.seed(seed)            
    m = X.shape[1]                  
    mini_batches = []        
   
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((10,m))
   
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
       
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]       
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
      
        mini_batch_X = shuffled_X[:,m-mini_batch_size*num_complete_minibatches:m]
        mini_batch_Y = shuffled_Y[:,m-mini_batch_size*num_complete_minibatches:m]       
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def initialize_velocity(parameters):
    L = len(parameters) //2
    v = {}
    
    for l in range(L):
        v["dW"+str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        v["db"+str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
        
    return v


def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    
    L = len(parameters)//2
    for l in range(L):
        v["dW"+str(l+1)] = beta*v["dW"+str(l+1)] + (1-beta)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta*v["db"+str(l+1)] + (1-beta)*grads["db"+str(l+1)]
        parameters["W" + str(l+1)] = parameters['W' + str(l+1)] - learning_rate*v['dW' + str(l+1)]
        parameters["b" + str(l+1)] = parameters['b' + str(l+1)] - learning_rate*v['db' + str(l+1)]
    
    return parameters ,v

def initialize_adam(parameters) :      
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
       
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    for l in range(L):
             
        v["dW" + str(l + 1)] =  beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
      
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))       
        
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
       
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
      
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * v_corrected["dW" + str(l + 1)]) / (np.sqrt(s["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate * v_corrected["db" + str(l + 1)]) / (np.sqrt(s["db" + str(l + 1)]) + epsilon)
      
    return parameters, v, s

def L_layer_model(X, Y, layers_dims,optimizer = "gd", learning_rate = 0.001, num_iterations = 5000, 
                  print_cost=False,keep_prob = 1,
                  mini_batch_size = 64 ,beta = 0.9,
                  beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):#lr was 0.009
   
    np.random.seed(1)
    costs = []        # keep track of cost   
    t = 0
    seed = 10                   
    parameters = initialize(layers_dims)   
    m = X.shape[1] 
    
    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "gd":
        None
    elif optimizer == "adam":          
        v , s = initialize_adam(parameters)    
        
    for i in range(0, num_iterations):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:
            (minibatch_X,minibatch_Y) = minibatch
            
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.       
            AL, caches,Ds =L_model_forward(minibatch_X,parameters,keep_prob)       
        
            # Compute cost.                  
          #  cost_total += compute_cost(AL,minibatch_Y)
            cost_total += compute_cost_with_regularization(AL,minibatch_Y, parameters)
            # Backward propagation.      
            grads = L_model_backward(AL,minibatch_Y,caches,Ds,keep_prob)    
 
            # Update parameters.     
            if optimizer =="gd":
                parameters = update_params(parameters,grads,learning_rate)
            elif optimizer == "momentum":
                parameters ,v = update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters ,v,s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
        
        cost_avg = cost_total / m
        # Print the cost every 100 training example
        if print_cost and i % 5 == 0:
          # print(cost)
            print ("Cost after iteration %i: %f" %(i, cost_avg))
        if print_cost and i % 5 == 0:
            costs.append(cost_avg)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


#Defining Layers of model
layers_dims = [X_train.shape[1], 256,256, 10]

#Building Model
parameters = L_layer_model(X_train.T, y_train.T, layers_dims, num_iterations = 50, print_cost = True,keep_prob = 0.6,mini_batch_size = 512,optimizer = "adam")

#Predicting On Training set
accuracy = predict(X_train.T,y_train.T,parameters)
print("Accuracy in Training Set " + str(accuracy))

#Predicting On Validation set
accuracy = predict(X_val.T,y_val.T,parameters)
print("Accuracy in validation Set " + str(accuracy))

#Predicting On Test set
accuracy ,y_hat= predict(X_test.T,y_test.T,parameters)
print("Accuracy in Test Set " + str(accuracy))

"""
Accuracy in Training Set (99.64848484848486)
Accuracy in validation Set (97.67619047619047)
Accuracy in Test Set 97.77

"""