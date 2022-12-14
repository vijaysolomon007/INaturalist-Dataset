import tensorflow as tf
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

Cnn = CNN(args['filters'],[(3,3),(3,3),(3,3),(3,3),(3,3)],[args['activation']]*6,(128,128,3),args['dense_layer'],10,weight_decay = args['weight_decay'],learning_rate = args['learning_rate'],data_augmentation = args['data_augmentation'],batch_size = args['batch_size'])

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filters", required=True,
	help="number of filters per layer")
ap.add_argument("-a", "--activation", required=True,
	help="activation function (relu,elu,selu)")
ap.add_argument("-dl", "--dense_layer", required=True,
	help="number of neurons in dense layer")
ap.add_argument("-wd", "--weight_decay", type=float, default=0,
	help="weight decay (L2 regulaiser)")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
	help="weight decay (L2 regulaiser)")

ap.add_argument("-bn", "--batch_normalization", type=bool, default=False,
	help="batch normalization (True or False)")

ap.add_argument("-da", "--data_augmentation", type=bool, default=False,
	help="threshold when applyong non-maxima suppression")

ap.add_argument("-bs", "--batch_size", type=int, default=64,
	help="batch size")

ap.add_argument("-do", "--dropout", type=int, default=64,
	help="batch size")
args = vars(ap.parse_args())

def ds_generation(data_augmentation):
 # test_datagen = ImageDataGenerator(rescale = 1./255)
  if not data_augmentation:
    train_datagen = ImageDataGenerator(rescale = 1./255,validation_split = 0.1)
  else:
    train_datagen = ImageDataGenerator(rescale = 1./255,
      shear_range = 0.1,
      rotation_range=20,
      width_shift_range=0.2,
      zoom_range = 0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      validation_split=0.1
      )
  train_ds = train_datagen.flow_from_directory(
    "/content/inaturalist_12K/train/",
    batch_size = 32,
    subset="training",
    target_size=(128, 128),
    class_mode='categorical'
  )
  val_ds =train_datagen.flow_from_directory(
    "/content/inaturalist_12K/train/",
    batch_size = 32,
    subset="validation",
    target_size=(128, 128),
    class_mode='categorical'
  )

  return train_ds,val_ds

train_da_ds, val_da_ds = ds_generation(True)
train_nda_ds, val_nda_ds = ds_generation(False)

test_datagen = ImageDataGenerator(rescale = 1./255)

test_ds =test_datagen.flow_from_directory(
    "/content/inaturalist_12K/val/",
    batch_size = 2000,
    target_size=(128, 128),
    class_mode='categorical'
  )

x_test,y_test = test_ds.next()

class CNN:

  def __init__(self,num_of_filters,size_of_filters,activation_function,input_shape,dense_layer_neurons,output_size,learning_rate = 0.0001,weight_decay = 0,batch_normalization = False,batch_size = 32,data_augmentation = False,dropout = 0):
    self.model = Sequential()


    self.batch_size = batch_size
    self.data_augmentation = data_augmentation
    self.learning_rate = learning_rate

    self.model.add(Conv2D(num_of_filters[0], size_of_filters[0],activation = activation_function[0],kernel_regularizer= l2(weight_decay),input_shape=input_shape)) 
    self.model.add(MaxPooling2D((2,2)))

    for i in range(4):

      if batch_normalization:
        self.model.add(BatchNormalization())

      self.model.add(Conv2D(num_of_filters[i+1], size_of_filters[i+1],activation = activation_function[i+1], kernel_regularizer= l2(weight_decay)))
      self.model.add(MaxPooling2D((2,2)))
      

    self.model.add(Flatten()) 
    self.model.add(Dense(dense_layer_neurons, activation=activation_function[-1], kernel_regularizer= l2(weight_decay)))

    if batch_normalization:
      self.model.add(BatchNormalization())

    if dropout:
      self.model.add(Dropout(rate = dropout))

    
      
    self.model.add(Dense(output_size, activation='softmax', kernel_regularizer= l2(weight_decay)))

  def Summary(self):
    print(self.model.summary())

  def Train(self):
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
    
    

    
    history = 0

    if self.data_augmentation:
      history = self.model.fit(train_da_ds,validation_data = val_da_ds,epochs = 20,batch_size = self.batch_size)
    else:
      history = self.model.fit(train_nda_ds,validation_data = val_nda_ds,epochs = 20,batch_size = self.batch_size)
    
    return history


  def Predict(self,x_test):
    y_pred = self.model.predict(x_test)
    return y_pred


Cnn = CNN(args['filters'],[(3,3),(3,3),(3,3),(3,3),(3,3)],[args['activation']]*6,(128,128,3),args['dense_layer'],10,weight_decay = args['weight_decay'],learning_rate = args['learning_rate'],data_augmentation = args['data_augmentation'],batch_size = args['batch_size'])

history = Cnn.Train()




y_pred = Cnn.Predict(x_test)
y_pred = y_pred.argmax(axis = 1)
y_test = y_test.argmax(axis = 1)

print("Train Accuracy:",history.history['accuracy'][-1])
print("Validation Accuracy:",history.history['val_accuracy'][-1])
print("Testing Accuracy:",(y_pred == y_test).mean())


fig = plt.figure(figsize=(60,50))

class_names = ['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

_, axs = plt.subplots(10, 3, figsize=(65, 65))
axs = axs.flatten()

label = 0
row = 0
for ax in axs:
    img = np.where(y_test == label)[0][(row+1)*10]

    ax.set_title("Actual : "+str(class_names[y_test[img]])+"\n Predicted : "+str(class_names[y_pred[img]]))
    ax.axis('off')
    ax.imshow(x_test[img])
    if row == 2:
      row = 0
      label+=1
    else:
      row += 1

plt.savefig("4b.png")

plt.show()

  
    
