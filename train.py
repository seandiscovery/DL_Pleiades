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

def fnn_train(fnn_model, X_train, X_val, Y_train, Y_val, X_weights, hyperParams, save=True, filepath="."):
    """
    Main training function for the FNN model. 
    Inputs: 
        - fnn_model: Object instance of the FNN class, as defined in models.py
        - X_train, Y_train, ..., X_weights: preprocessed data outputs of the function 
            fnn_input, from models.py 
        - hyperParams: a list containing the desired hyperparameters for the Adam optimizer. 
          We assume the format: 

                hyperParams = [learning_rate, epsilon, decay_rate, number_epochs, batch_size]

          See the Keras optimizer documentation for more info 

    Outputs: 
        - history: a Keras History object, to store the metrics of the model during training 
    If the flag "save" is set to true, a full copy of the model is saved at the indicated filepath
    """
    # extract the keras model object stored inside fnn_model 
    model = fnn_model.return_model() 
    # define the optimizer 
    Adam = optimizers.Adam(lr=hyperParams[0], epsilon=hyperParams[1], decay=hyperParams[2]) 
    # compile the model 
    model.compile(Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # train the model!     
    model_history = model.fit(X_train, Y_train, batch_size=hyperParams[4], epochs=hyperParams[3], 
                    validation_data=(X_val, Y_val), shuffle=True, sample_weight=X_weights)

    if save: 
        model.save(filepath)

    return model_history




