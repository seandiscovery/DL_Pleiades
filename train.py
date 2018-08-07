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
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
    model_history = model.fit(X_train, Y_train, batch_size=hyperParams[2], epochs=hyperParams[1], validation_data=(X_val, Y_val), callbacks=[EarlyStopping(verbose=True, patience=25, monitor='val_loss'), ModelCheckpoint(filepath, verbose=1, save_best_only=True)], shuffle=True, sample_weight=X_weights)
    
    """
    Deprecated in favor of Callback best-model saving, 08/01 
    if save: 
        model.save(filepath)
    """

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

def ete_train(ete_model, rnn_X_train, rnn_Y_train, rnn_weights, dnn_X_train, dnn_Y_train, dnn_X_val, dnn_Y_val, dnn_weights, 
    hyperParams=[1e-3,50,50], rnn_val_split=0.2, save=True, filepath="ete_model", default=False):
    """
    Main training function for the end-to-end (ete) model. Inputs correspond to the inputs of dnn_train and rnn_train, with 
    the appropriate qualifier ("rnn_" or "dnn_") in front of the variable name. 
    """
    # RNN: check for an embedding 
    if ete_model.rnn_embed: 
        # important note: the first element of this list corresponds to the continuous track 
        # inputs (e.g. S_d0, S_z0), while the second element is the track grade (category) 
        rnn_X_train = [rnn_X_train[:,:,:-1], rnn_X_train[:,:,-1]]
        print(rnn_X_train[0].shape) 
        print(rnn_X_train[1].shape) 
        
    # NOTE: code not structured for rnn_embed = False --> fix this!  

    # RNN: select validation set according to 0 < rnn_val_split < 1 
    # Step 1: generate array of indices for rnn_X_train, randomly sample 
    rnn_size = rnn_Y_train.shape[0]
    X_train_indices = np.arange(rnn_size, dtype=int)
    random_indices = np.random.choice(X_train_indices, size=int(rnn_size*rnn_val_split), replace=False)
    # Step 2: select data from rnn_X_train, rnn_Y_train using numpy smart indexing 
    cont_track_val = rnn_X_train[0][random_indices]
    grade_track_val = rnn_X_train[1][random_indices]
    
    print(cont_track_val.shape) 
    print(grade_track_val.shape) 
    
    rnn_Y_val = rnn_Y_train[random_indices]
    # Step 3: remove the validation set from the training set 
    cont_track_train = np.delete(rnn_X_train[0], random_indices, axis=0) 
    grade_track_train = np.delete(rnn_X_train[1], random_indices, axis=0)
    rnn_Y_train = np.delete(rnn_Y_train, random_indices)
    
    print(cont_track_train.shape) 
    print(grade_track_train.shape) 
    
    # extract the keras model object stored inside ete_model 
    model = ete_model.return_model()
    # define the optimizer 
    if default:
        Adam = optimizers.Adam()
    else: 
        Adam = optimizers.Adam(lr=hyperParams[0])
    # compile the model 
    model.compile(Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # train the model!  
    # running without weights, for the purposes of testing --> how do we add weighting for multiple inputs? 
    model_history = model.fit([cont_track_train, grade_track_train, dnn_X_train], [rnn_Y_train, dnn_Y_train], batch_size=hyperParams[2], epochs=hyperParams[1], validation_data=[(cont_track_val, grade_track_val, rnn_Y_val), (dnn_X_val, dnn_Y_val)], shuffle=True)

    if save: 
        model.save(filepath)