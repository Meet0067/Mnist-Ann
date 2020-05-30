# Mnist A Digit Classifier Using Artificial Neural Network

1) digit_Classifier_Using_Keras.py
      
        Train set : 99.95 %
        Validation set: 98.2 %
        test set: 98.24 %
        
2) digit_Classifier_Scratch.py
    
        Train set : 99.65 %
        Validation set : 97.7 %
        Test set : 97.8 %
        
      As you can see the power of libraries.I implemented digit_Classifier_Using_keras.py from Scratch.
      Also with this file ,you can check Accuracies with hyperparameter tuning.
      Like ,if you want an optimizer gradient descent  then simply change the arguments of Buiding Model by
            optimizer  = "gd"
           ,if you want adam then
            optimizer = "adam" 
            ,and if you want momentum then
            optimier = "momentum".
            
      Here the term Keep_prob = 0.6 means DropOut,
            That means if keep_prob = 0.6 then 40 % of nodes are dropped in the layers during training of the model .
            If you set keet_prob = 0.8 then 20 % of nodes are dropped in your layers during the training of the model .
            FYI => Dropout technique is used to prevent overfitting in the model.So it is not applied during test set.
            
      Also you can tune batch size , learning rate ,epochs and beta.
            
