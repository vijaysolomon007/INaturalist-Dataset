A Deep learning programming assignment where we have to develop a Convolution Neural Network from scratch to classify the images in INATURALIST dataset

## Functions 
#### CNN():
- num_of_filters : number of filters per each layer 
- size_of_filters : Size of Each filter
- activation_function : Activation function for each convolution layer (relu or elu or selu)
- input_shape : Input image Shape 
- dense_layer_neurons : number of Neurons in Dense layer (Pre Output Layer)
- output_size : Number of Classes present in the data
- learning_rate : Learning rate of the network (default = 0.0001)
- weight_decay : weight decay value for L2 Regularization (default = 0)
- batch_normalization: bool value that whether batch normalization layer should be added or not (default = False)
- batch_size : batch size for data (default = 32)
- data_augmentation : bool value that whether data augmentation should be applied or not (default = False)
- dropout : dropout rate value for the network (default : 0)

#### Train():
 Compiles the CNN model and trains the input data and gives accuracies for validation data
 
#### Predict():
  Predicts the output on test data
  
## Hyperparameter configurations used
 - weight decay: 0,0.0005,0.00005
 - dropout: 0,0.1,0.2,0.4
 - learning_rate:  0.0001,0.00001
 - activation: 'relu','elu','selu'
 - batch_norm: False,True 
 - filters:  [[32,32,32,32,32],[128,64,64,32,32],[32,64,128,256,512]]
 - data_augmentation: False,True
 - batch_size: 32,64
 - dense_layer: 64,128,256,512,1024
  
#### Training the model
 - to train the model we need to initalise the Neural network with CNN() function and Train the model using Train() method it will display validation accuracy, training accuracy 
#### Evaluating the model
 - to evaluate the model just pass the test data to Predict() function it will return the predicted class labels on test data and use evaluate() method for metrics
