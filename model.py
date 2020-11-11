# %% Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from data_analysis import load_data
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2, l1, l1_l2

# %% Loading the data by using the "load_data" function from data_analysis.py file

(x_train, y_train), (x_test, y_test) = load_data(test_size= 0.2, n_mfcc= 74)


# %% Defining models to be tried

def rnn_model():
    i = layers.Input(x_train.shape[1:])
    x = layers.Masking(mask_value= 0)(i)
    x = layers.Bidirectional(layers.LSTM(64, kernel_regularizer= l2(0.001),return_sequences= True))(x)
    x = layers.Bidirectional(layers.LSTM(32, kernel_regularizer= l2(0.001)))(x)
    x = layers.Dense(64, activation= 'relu', kernel_regularizer= l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation= 'relu', kernel_regularizer= l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation= 'sigmoid')(x)
    
    return Model(inputs= i, outputs= out)

def conv2D_model():
    
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=x_train.shape[1:],
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    tf.keras.layers.Dropout(0.5)
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    tf.keras.layers.Dropout(0.5)

    # softmax output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model

def conv1D_model():    

    model = Sequential()
    
    model.add(layers.Conv1D(128, 3, padding='same', input_shape=(x_train.shape[1:])))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(64, 3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(32, 3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.GRU(16, activation= 'tanh', return_sequences= True))

    model.add(layers.GRU(16, activation= 'tanh'))

    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))
    
    return model

# %% Defining our model

# Getting an instance of the rnn model defined above
model = rnn_model()

# Printing the model's summary
model.summary()

# Defining some useful callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_accuracy', verbose= 1, patience= 5,  restore_best_weights=True)
checkpoint = ModelCheckpoint('checkpoint', monitor= 'val_accuracy', verbose= 1,
                             save_best_only= True, save_freq= 'epoch', save_weights_only= False)

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer = optimizers.Adam(lr=0.001), metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train, batch_size= 32, epochs= 50, validation_data= (x_test, y_test),
                    callbacks=[reduce_lr, early_stop, checkpoint])


# %% Accuracy of the model after restoring the best weights
print(model.evaluate(x_test, y_test))


# %% Plotting the model's accuracy and loss
def plot_history(history):

    fig, axs = plt.subplots(2, 1, figsize= (10, 7))

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="Training Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracies")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="Training Loss")
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Losses")

    plt.show()
    

plot_history(history)
