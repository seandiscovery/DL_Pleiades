"""
This script allows for multiple models to be trained (and outputs saved) by running only this file 
from the terminal command line. Initially, we will provide functionality for basic hyperparameter 
tuning (e.g. learning rate, number of hidden layers), but will add functionality as training progresses, 
or as requested. 

An argparser is included, so the user can input the preprocessed data files from the command line. Note that 
the argparser does not yet include support for inputing files to the RNN training script.  

Sean Mullane, August 2018 
"""

from __future__ import division 
from train import * 
import models 
import numpy as np 
import h5py

### Convenient global functions ### 

def history_save(hist_object, model_num, layers_num, hist_save_path=""):
    """
    Convenience function to save a history object returned by (r/f)nn_train; code courtesy 
    Nicole Hartman, trainNet.py
    Input: 
        - hist_object: history object returned by the Keras fit function 
        - model_num, layers_num: model ID number and model's number of hidden layers, 
                                 for decriptive naming 
        - hist_save_path: folder path to save history (note that model name is defined in 
            this function)
    """
    # construct empty arrays to holds various history params 
    loss, val_loss = np.array([]), np.array([])  
    acc,  val_acc  = np.array([]), np.array([])
    
    full_path = hist_save_path + "{}_{}_history_fnn.hdf5".format(model_num, layers_num)
    h = h5py.File(full_path, 'w')

    h.create_dataset("loss",     data=np.append(loss,    hist_object.history['loss']))
    h.create_dataset("val_loss", data=np.append(val_loss,hist_object.history['val_loss']))
    h.create_dataset("acc",      data=np.append(acc,     hist_object.history['acc']))
    h.create_dataset("val_acc",  data=np.append(val_acc, hist_object.history['val_acc']))
    print("Saving history object")
    h.close() 

###### FNN Training #######

def fnn_train_script(train_file, val_file, weight_file, num_classes=3, verbose=True):
    """
    Main FNN training script. 
    Explicit function inputs: 
        - train_file, val_fule, weight_file: preprocessed training, validation, weight data. 
            See the docstring of fnn_input, in models.py, for more information. 
        - num_classes: Number of possible classes. Default is three for DL1. 
        - verbose: whether to print descriptive updates during training 

    Implicit funciton inputs: 
        - hidden_layers: list holding the total number of hidden layers for each model. Note that 
            len(hidden_layers) == total # of models trained 
        - model_dimensions: dictionary with an integer key corresponding to the model's index in 
            hidden_layers, and a value of a list holding the hidden layer dimensions of the model 
        - model_hyper: dictionary with same integer key as model_dimensions, value a list holding 
            [learning_rate, number_epochs, batch_size]

    Model saves model objects, history objects to desired folder. 
    """
    # extract the input vectors 
    X_train, X_val, Y_train, Y_val, X_weights = models.fnn_input(train_file, val_file, weight_file)
    num_vars = X_train.shape[1]
    if verbose: 
        print("-----Extracting input vectors-----")

    # define the path to the necessary results folder 
    model_save_path = "Results/FNN_models/"
    hist_save_path = "Results/FNN_histories/"

    ### define necessary hyperparameters ###

    hidden_layers = [3] # list holding total number of hidden layers for each model 
    num_models = len(hidden_layers) # number of models to train

    # construct a dictionary that holds the number of dimensions per hidden layer of each model 
    """
    Assumed format: key == 'i', where i is the index of the model in the array hidden_layers. 
    Value == list of integers, where len(Value) == hidden_layers[i] (e.g. list of dimensions must be same 
    size as number of hidden layers). 
    """
    model_dimensions = {}
    model_dimensions['0'] = [150, 100, 50]

    # construct a dictionary that holds the basic hyperparameters for each model
    """
    Assumed format: key == 'i', as above. Value == list of integers: [learning_rate, number_epochs, batch_size]
    Note that the Value corresponds to the array hyperParams in fnn_train, train.py 
    """
    model_hyper = {}
    model_hyper['0'] = [1e-3, 50, 50]

    ### main training loop ### 
    if verbose: 
        print("-----Constructed hyperparams, beginning main training loop-----")
    for i in range(num_models): 
        # step 0: name the model appropriately 
        """
        Naming convention: the first number is the model index in hidden_layers, followed by the number of hidden layers, 
        followed by a descriptive name. Ex: 0_4_fnnModel
        """
        model_string = "{}_{}_fnnModel.h5".format(i, hidden_layers[i])

        # step 1 : define the model 
        model_i = models.FNN(input_dim=num_vars, num_hidden=hidden_layers[i], hidden_dims=model_dimensions[str(i)], 
            modelName=model_string)

        # step 2: train the model 
        if verbose: 
            print("Training model " + str(i))
        full_save = model_save_path + model_string
        history_i = fnn_train(model_i, X_train, X_val, Y_train, Y_val, X_weights, hyperParams=model_hyper[str(i)], 
            filepath=full_save)

        # step 3: save the history 
        history_save(history_i, i, hidden_layers[i], hist_save_path)

# https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == "__main__":

    from argparse import ArgumentParser

    p = ArgumentParser()

    # load input files, from the command line 
    p.add_argument("--train_file", type=str, default="", dest="train_file", help="Preprocessed FNN training file (see fnn_input in models.py")
    p.add_argument("--val_file", type=str, default="", dest="val_file", help="Preprocessed FNN validation file")
    p.add_argument("--weight_file", type=str, default="", dest="weight_file", help="Preporcessed FNN weight file")
    p.add_argument("--verbose", type=bool, default="True", dest="flag", help="Print updates during training")
    
    args = p.parse_args()

    fnn_train_script(args.train_file, args.val_file, args.weight_file, verbose=args.flag)
 
    print("Successfully trained all models!")
         


