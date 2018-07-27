"""
Unified file for three main b-tagging algorithms: 
    1) Fully connected feedforward neural network, trained on the 
        DL1 variables. 
    2) Recurrent neural network, originally written by Nicole Hartman;
        provided here without edits for completness. 
    3) An end-to-end (ETE) algorithm that combines the RNN and FNN into 
        a single neural network.

The algorithms are wrapped in classes to make experimentation easier from an 
iPython notebook. However, we also include an argparser to construct the models directly 
from the command line. 

RNN: Nicole Hartman, Winter - Summer 2018 
FNN: Sean Mullane, with input and preprocessing steps (see GitHub repository) taken from DL1, 
        produced by the ATLAS flavor tagging group, CERN; inspiration also from Nicole Hartman's
        RNN code, CS231N project code (Nicole Hartman + Sean Mullane, Spring 2018)
ETE: Sean Mullane, Summer 2018 
"""
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Input, Dense, Activation, Dropout, BatchNormalization
import h5py
from keras.utils.np_utils import to_categorical

####### Useful global functions ########

def fnn_input(train_file, val_file, weight_file, num_classes):
    """
    Adopted from the DL1 repository's BatchTrainModel.py. 
    Input: 
        - train_file: preprocessed training data, output of preprocessing_fnn.py with 
            --add_dl1 == 0 
        - val_file: preprocessed validation data, output of preprocessing_fnn.py with 
            --add_dl1 == 1
        - weight_file: preprocessed weight file, output of weightComputation_fnn.py 

    Output: The training vector X_train, the validation vector X_val, and the loss weight 
                vector X_weights; also the ground truth labels Y_train and Y_val 
    """
        
    # open the files 
    trainF = h5py.File(train_file, 'r')
    valF = h5py.File(val_file, 'r') 
    wF = h5py.File(weight_file, 'r')  

    # extract training data for b, c, u jets 
    train_bjet = trainF['train_processed_bjets'][:]
    train_cjet = trainF['train_processed_cjets'][:]
    train_ujet = trainF['train_processed_ujets'][:]
    # extract validation data 
    val_bjet = valF['train_processed_bjets'][:]  
    val_cjet = valF['train_processed_cjets'][:]
    val_ujet = valF['train_processed_ujets'][:]

    # configure the variables which are used for the training
    var_names = list(train_bjet.dtype.names)
    var_names.remove('label')

    # Note: at this point in BatchTrainModel, there is a function to remove variables 
    # used for training. However, the list it selects from var_remove is always empty, 
    # from what I can tell. So I'm excluding this for now. 

    # extract weight files 
    c_weight = wF['c_vs_b_weights'][:]
    u_weight = wF['u_vs_b_weights'][:]
    b_weight = wF['b_vs_b_weights'][:]

    # construct train, validation vectors 
    X_train = np.concatenate( ( train_ujet, train_cjet, train_bjet ) )  
    X_val  = np.concatenate( ( val_ujet,  val_cjet,  val_bjet  ) )

    # construct ground truth labels 
    y_train = np.concatenate( ( np.zeros(len(train_ujet)), np.ones(len(train_cjet)), 2*np.ones(len(train_bjet)) ) )
    y_val  = np.concatenate( ( np.zeros(len(val_ujet)), np.ones(len(val_cjet)), 2*np.ones(len(val_bjet)) ) )

    # prepare ground truth labels for categorical cross-entropy loss 
    # see Keras loss functions documentation 
    Y_train = to_categorical(y_train, num_classes)   
    Y_val  = to_categorical(y_val, num_classes)

    #### Construct the input vectors #### 
    X_train = X_train[var_names]
    X_train = np.array(X_train.tolist())
    X_val = X_val[var_names]
    X_val = np.array(X_val.tolist())

    # construct the loss weights vector 
    X_weights = np.concatenate ( (u_weight, c_weight, b_weight) )

    # after this, BatchTrainModel shuffles the data; however, I think 
    # that we can get Keras to do this for us, so I'm excluding it for now 

    return X_train, X_val, Y_train, Y_val, X_weights

######## Class Definitions ########## 
# Classes that define the various models 

class FNN():

    def __init__(self, input_dim=46, num_hidden=3, hidden_dims=[150, 100, 50], out_dim=3, p=0.5, modelName="FNN_model"):
        """
        Constructor for the FNN. 
        Inputs: 
            - input_dim: Dimensionality of the input vector. Default of 46 is for standard DL1 input parameters. 
            - num_hidden: Number of hidden layers. Default: 3 
            - hidden_dims: List containing the dimension of each hidden layer. NOTE: len(hidden_dims) == num_hidden, or 
                the model will not be constructed. Default: 150 --> 100 --> 50 
            - out_dim: Output dimension for the final Softmax layer. Default: 3
            - p: Probability of dropout. Default: 0.5 
            - modelName: Name of model for saving, if necessary 
        """
        self.inDim = input_dim 
        self.hidden = num_hidden

        # construct dictionary to hold hidden dimensions 
        # Ex: hidden_layers['2'] = number of dimensions in second hidden layer 
        hidden_layers = {}
        for i in range(len(hidden_dims)):
            hidden_layers[str(i)] = hidden_dims[i] 

        self.outDim = out_dim 
        self.p = p
        self.modelName = modelName

        ### construction of the model ###

        # input vector
        """
        A weird keras note here: the number of input examples is NOT included in the shape of the 
        input vector. E.g., even though we have ~1000 input jets, the shape of the input is still considered 
        to be one-dimensional (46,). Otherwise, this throws a dimension mismatch
        """
        fnn_vector = Input(shape=(self.inDim,), dtype='float')

        """
        Possible issue: batchnorm after activation? Keras doesn't make it easy 
        to use batchnorm inbetween the linear and non-linear layers 
        """ 
        # define the model architecture 
        # generalized for an arbitrary number of hidden layers 
        for j in range(self.hidden): 
            if j == 0: 
                # initialization 
                x = Dense(hidden_layers['0'], activation='relu')(fnn_vector)
                x = BatchNormalization(axis=1)(x)
                x = Dropout(rate=self.p)(x) 

            else: 
                # hidden layers 
                x = Dense(hidden_layers[str(j)], activation='relu')(prev_x)
                x = BatchNormalization(axis=1)(x)
                x = Dropout(rate=self.p)(x) 
        
            # store previous output 
            prev_x = x 

        # final softmax layer 
        results = Dense(out_dim, activation='softmax')(prev_x)

        # define the model
        self.model = Model(inputs=fnn_vector, outputs=results)
        
    def return_model(self): 
        # convenience function that returns the model constructed above
        # to avoid having to call myModel.model every time the model needs to be used 
        return self.model 



