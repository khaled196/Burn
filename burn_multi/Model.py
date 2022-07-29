import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tensorflow.keras.callbacks import TensorBoard
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

opt = tf.keras.optimizers.Adam(learning_rate=0.001)


NAME = "Burn-CNN-{}".format(int(time.time()))

tensorboard=TensorBoard(log_dir="logs/{}".format(NAME))

# IN-DIR
os.chdir(sys.argv[1])
data= os.path.join(sys.argv[1],'data')

pickle_in = open(os.path.join(data,"x.pickle"),"rb")
x = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"y.pickle"),"rb")
y = pickle.load(pickle_in)




def CNN_model():
  k = 4
  num_val_samples = len(x) // k
  num_epochs = 20
  all_scores = []
  history_list=[]
  
  for i in range(k):
      print('processing fold #', i)
      val_data = x[i * num_val_samples: (i + 1) * num_val_samples]
      val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
      partial_train_data = np.concatenate(
      [x[:i * num_val_samples],
      x[(i + 1) * num_val_samples:]],
          axis=0)
      partial_train_targets = np.concatenate(
      [y[:i * num_val_samples],
      y[(i + 1) * num_val_samples:]],
      axis=0)
      model = Sequential()
      model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:]))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      #model.add(MaxPooling2D(pool_size=(2, 2)))

  
      model.add(Conv2D(64, (3, 3)))
      model.add(Activation('relu'))
      model.add(ZeroPadding2D(padding=(1,1)))
      model.add(BatchNormalization())

      #model.add(MaxPooling2D(pool_size=(2, 2)))
      
      model.add(Conv2D(128, (3, 3), input_shape=x.shape[1:]))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
  
      model.add(Conv2D(128, (3, 3)))
      model.add(Activation('relu'))
      model.add(ZeroPadding2D(padding=(1,1)))
      model.add(BatchNormalization())
      model.add(AveragePooling2D(pool_size=(2, 2)))
  
      model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
  
      model.add(Dense(64))
      model.add(Activation('relu'))
  
      model.add(Dense(32))
      model.add(Activation('relu'))
  
      model.add(Dropout(0.1))
  
      model.add(Dense(3))
      model.add(Activation('softmax'))
  
      model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
  
      history= model.fit(partial_train_data, partial_train_targets,
      epochs=num_epochs, batch_size=64, verbose=0, callbacks=[tensorboard],validation_data=(val_data, val_targets))
      history_list.append(history.history)
      #print(history_list)
      #val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
      #all_scores.append(val_mae)
      #print(np.mean(all_scores))
      model.save('Burn_images.model')
  return history_list

history_list = CNN_model()




#locate where the output will be saved in
folder= os.path.join(sys.argv[1],'output')
if not os.path.exists(folder):
  os.mkdir(folder)

count=0     
for history in history_list:
  count=count+1
  accuracy = history['accuracy'] 
  print (accuracy)
  val_accuracy = history['val_accuracy']
  print (val_accuracy)
  loss = history['loss'] 
  print(loss)
  val_loss = history['val_loss'] 
  print(val_loss)
  epochs = range(len(accuracy))
  print(epochs) 
  plt.plot(epochs, accuracy, 'bo', label='Training accuracy') 
  plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy') 
  plt.title('Training and validation accuracy') 
  plt.legend() 
  plt.figure() 
  plt.plot(epochs, loss, 'bo', label='Training loss') 
  plt.plot(epochs, val_loss, 'b', label='Validation loss') 
  plt.title('Training and validation loss') 
  plt.legend() 
  plt.savefig(os.path.join(folder,'Compare training and validation results for K fold set number {}'.format(count)))

