"""
Main training file for the algorithms defined in models.py. 
For now, each algorithm (FNN, RNN, ETE) has its own training function, since 
the algorithms were made by different developers. Future development will 
attempt to construct a general training function for all three algorithms. 

Sean Mullane Summer 2018, with code from DL1 (ATLAS flavor tagging group) and Nicole 
Hartman's RNN code, as cited. 
"""

from models import FNN # needed? 
from keras.models import Model
from keras import optimizers 
import h5py
import numpy as np 

def fnn_train(fnn_model, X_train, X_val, Y_train, Y_val, X_weights, hyperParams=[1e-3,50,50], save=True, filepath="./fnn_model", default=False):
    """
    Main training function for the FNN model. 
    Inputs: 
        - fnn_model: Object instance of the FNN class, as defined in models.py
        - X_train, Y_train, ..., X_weights: preprocessed data outputs of the function 
            fnn_input, from models.py 
        - hyperParams: a list containing the desired hyperparameters for the Adam optimizer. 
          We assume the format: 

                hyperParams = [learning_rate, number_epochs, batch_size]

          See the Keras optimizer documentation for more info 
        - default: temporary flag for training with all default Adam parameters (e.g. lr = 0.001), or 
                    putting in learning rate manually. There seems to be some difference in performance 
                    between putting in a manual learning rate of lr=1e-3, and just using the default (which 
                    also sets a learning rate of 1e-3). 

    Outputs: 
        - history: a Keras History object, to store the metrics of the model during training 
    If the flag "save" is set to true, a full copy of the model is saved at the indicated filepath
    """
    # extract the keras model object stored inside fnn_model 
    model = fnn_model.return_model() 
    # define the optimizer 
    if default: 
        Adam = optimizers.Adam()
    else:
        Adam = optimizers.Adam(lr=hyperParams[0]) 
    # compile the model 
    model.compile(Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # train the model!   
    model_history = model.fit(X_train, Y_train, batch_size=hyperParams[2], epochs=hyperParams[1], 
                              validation_data=(X_val, Y_val), shuffle=True, sample_weight=X_weights)

    if save: 
        model.save(filepath)

    return model_history

def rnn_train(rnn_model, X_train, y_train_cat, weights_train, hyperParams=[1e-3,50,50], val_split=0.2, save=True, filepath="./rnn_model", default=False):
    """
    Main training function for the RNN LSTM model. Adopted from Nicole Hartman's code in trainNet.py 
    Inputs: 
        - rnn_model: Object instance of the LSTM class, as defined in models.py 
        - X_train, y_train_cat, weights_train: preprocessed data vector outputs of the function 
            rnn_input(_simple), from models.py 
        - hyperParams: a list containing esired hyperparameters for the Adam optimizer 
            We assume the same format as in the function fnn_train above. 
        - val_split: validation split to use 
        - default: temporary flag for training with all default Adam parameters (e.g. lr = 0.001), or 
                    putting in learning rate manually. There seems to be some difference in performance 
                    between putting in a manual learning rate of lr=1e-3, and just using the default (which 
                    also sets a learning rate of 1e-3). 
    Outputs: 
        - history: a Keras History object 
    As in fnn_train, model object saved to desired filepath if the flag "save" is true 
    """
    # check for an embedding 
    if rnn_model.embed: 
        X_train = [X_train[:,:,:-1],X_train[:,:,-1]]

    # extract the keras model object stored inside fnn_model 
    model = rnn_model.return_model() 
    # define the optimizer 
    if default:
        Adam = optimizers.Adam()
    else: 
        Adam = optimizers.Adam(lr=hyperParams[0])
    # compile the model 
    model.compile(Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # train the model!     
    model_history = model.fit(X_train, y_train_cat, batch_size=hyperParams[2], validation_split=val_split, epochs=hyperParams[1], 
                    shuffle=True, sample_weight=weights_train)

    if save: 
        model.save(filepath)

    return model_history
