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
from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Masking, Embedding, Flatten, Input, Dense, LSTM, concatenate, Activation, Dropout, BatchNormalization
import h5py
from keras.utils.np_utils import to_categorical

####### Useful global functions ########

def fnn_input(train_file, val_file, weight_file, num_classes=3):
    """
    Adopted from the DL1 repository's BatchTrainModel.py. 
    Input: 
        - train_file: preprocessed training data, output of preprocessing_fnn.py with 
            --add_dl1 == 0 
        - val_file: preprocessed validation data, output of preprocessing_fnn.py with 
            --add_dl1 == 1
        - weight_file: preprocessed weight file, output of weightComputation_fnn.py 
        - num_classes: total number of output classes (default for DL1 is 3) 

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

def rnn_input_simple(input_file, nJets=5e6):
    """
    Constructs input vectors for the LSTM RNN, without functionality to select 
    only a subset of input features, etc. Assumes preprocessed input file.   
    Adopted from Nicole Hartman's code in trainNet.py. 
    Inputs: 
        - input_file: preprocessed hdf5 file, usually output from processNpArray.py 
        - nJets: number of jets to use during training. If the number of jets is < default 
                 of 5e6, the model will train on a reduced number of training examples  
    """ 
    # open the file 
    f = h5py.File(input_file,"r")

    # extract relevant vectors 
    X_train       = f['X_train'][:]
    y_train       = f['y_train'][:]
    ix_train      = f['ix_train'][:]
    weights_train = f['weights_train'][:]

    # truncate the array if nJets < 5e6 
    """
    Depracated for testing with ~1k jets 07/30/18 
    if nJets < 5e6: 
        nTrain = int(X_train.shape[0] * nJets / 5e6)

        X_train       = X_train[:nTrain]
        y_train       = y_train[:nTrain]
        ix_train      = ix_train[:nTrain]
        weights_train = weights_train[:nTrain]
    """

    # prepare ground truth labels for categorical cross-entropy 
    y_train_cat = to_categorical(y_train, num_classes=4)

    return X_train, y_train_cat, ix_train, weights_train 

def processInputs(timeSteps, nFeatures, num_categories=14):
    """
    Function to process track inputs when doing categorical embedding. 
    Written by Nicole Hartman, Winter - Spring 2018; from trainNet.py. 
    Note: Default for num_categories taken from the global variable N_CATEGORIES 
    in the file trainNet.py 
    """
    cts_trk_inputs = Input(shape=(timeSteps, nFeatures - 1), name="Continuous_trk_inputs")
    masked_input = Masking()(cts_trk_inputs)

    # Next, look at a 2d embedding layer
    grade_input = Input(shape=(timeSteps,), name="Categorical_trk_inputs", dtype='int32')
    embed = Embedding(num_categories + 1, 2, mask_zero=True, input_length=timeSteps)(grade_input)

    # Merge the two layers
    myMerge = concatenate([masked_input, embed])

    trkEmbeddedInputs = Model(inputs=[cts_trk_inputs, grade_input], outputs=myMerge)
    return trkEmbeddedInputs

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
                """
                Depracated for initial model development 07/31 
                x = BatchNormalization(axis=1)(x)
                x = Dropout(rate=self.p)(x) 
                """

            else: 
                # hidden layers 
                # batchnorm included for second layer (three layer network test 08/02) 
                x = Dense(hidden_layers[str(j)], activation='relu')(prev_x)
                if j == 1: 
                    x = BatchNormalization(axis=1)(x)
                
                #x = Dropout(rate=self.p)(x) 
        
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

class LSTM_class(): 

    def __init__(self, num_hidden, num_dense, doEmbedding=True, nJets=5e6, maxNumTrks=15, 
                 timeSteps=15, nFeatures=5, p=0.2, modelName="RNN_model"):
        """
        Class for constructing an LSTM RNN model. This code is taken from trainNet.py, written by 
        Nicole Hartman (SLAC-ATLAS) Winter - Spring 2018, with some minor adjustments/simplifications 
        to integrate with the DL Pleiades workflow. 

        Inputs: 
            - num_hidden: number of hidden states for the RNN model (e.g. LSTM)
            - num_dense: number of dense layers for the FNN following the RNN model
            - doEmbedding: boolean flag; if True, track category is embedded in input vector (?) 
            - nJets: number of jets in the dataset
            - maxNumTrks: maximum number of tracks per jet to process 
            - timeSteps: number of timesteps for the RNN 
                NOTE: I believe maxNumtrks == timeSteps 
            - nFeatures: number of input features 
            - p: dropout probability 
            - modelName: a model name, if necessary 
        """
        self.hidden = num_hidden
        self.dense = num_dense
        self.embed = doEmbedding
        self.nJets = nJets
        self.maxTrks = maxNumTrks
        self.timeSteps = timeSteps
        self.features = nFeatures
        self.p = p 
        self.name = modelName

        num_classes = 4 # from global variable N_CLASSES, trainNet.py 

        # construct embedding, if necessary 
        if self.embed: 
            cts_trk_inputs = Input(shape=(self.timeSteps, self.features - 1), name="Continuous_trk_inputs")
            grade_input = Input(shape=(self.timeSteps,), name="Categorical_trk_inputs", dtype='int32')

            trk_inputs=[cts_trk_inputs, grade_input]

            # note that the global function processInputs returns a model object 
            # this is why it's passed trk_inputs 
            masked_input = processInputs(self.timeSteps, self.features)(trk_inputs)

        else: 
            trk_inputs = Input(shape=(self.timeSteps, self.features), name="Trk_inputs")
            masked_input = Masking()(trk_inputs)

        # Feed this merged layer to an RNN
        lstm = LSTM(self.hidden, return_sequences=False)(masked_input)  
        dpt = Dropout(0.2)(lstm)

        # Fully connected layer
        FC = Dense(self.dense, activation='relu')(dpt)

        # Softmax for classification
        output = Dense(num_classes, activation='softmax')(FC)
        self.model = Model(inputs=trk_inputs, outputs=output)

    def return_model(self): 
        # convenience function that returns the model constructed above
        # to avoid having to call myModel.model every time the model needs to be used 
        return self.model
    
class ETE():

    def __init__(self, dnn_input_dim, dnn_num_hidden=3, dnn_hidden_dims=[150, 100, 50], dnn_out_dim=3, dnn_p=0.5, 
        rnn_hidden=50, rnn_dim_dense=10, rnn_doEmbedding=True, rnn_timeSteps=15, rnn_Features=5, rnn_p=0.2, fnn_input_dim=46):
        """
        End-to-end (ETE) model construction. See the repository Wiki for an overview of the structure of this model.
        The rnn variables are tagged with "rnn_", while the quantities tagged "dnn_" tag the parameters of the deep 
        neural network. This DNN trains on the concatenated output of the RNN and the 46-dimensional input vector of DL1.
        The variable fnn_input_dim gives the dimensionality of the DL1 input vector (defaults to 46 dimensions).
        Note that we currently do not have a FNN preprocessing layer for the 46-dimensional DL1 input, although this 
        might be added in later testing (eg to reduce dimensionality of this vector). 

        In the current model, we do the RNN processing and treat the DL1 input vector as an "auxillary input", following 
        the example of the Keras functional API tutorial: https://keras.io/getting-started/functional-api-guide/
        """
        # lone DL1 parameter 
        self.dl_input = fnn_input_dim

        ## DNN parameters ## 
        self.dnn_input = dnn_input_dim
        self.dnn_hidden = dnn_num_hidden
        self.dnn_out = dnn_out_dim
        self.dnn_p = dnn_p 

        # construct dictionary to hold DNN hidden dimensions, as in the class FNN above  
        # Ex: hidden_layers['2'] = number of dimensions in second hidden layer 
        dnn_hidden_layers = {}
        for i in range(len(dnn_hidden_dims)):
            dnn_hidden_layers[str(i)] = dnn_hidden_dims[i] 

        ## RNN parameters ## 
        self.rnn_hidden = rnn_hidden
        self.rnn_dense = rnn_dim_dense
        self.rnn_embed = rnn_doEmbedding
        self.rnn_steps = rnn_timeSteps
        self.rnn_feat = rnn_Features
        self.rnn_p = rnn_p 

        ##### ETE Model Construction ##### 

        """ ------RNN processing------- """

        num_classes = 4 # from global variable N_CLASSES, trainNet.py 

        # construct embedding, if necessary 
        if self.rnn_embed: 
            cts_trk_inputs = Input(shape=(self.rnn_steps, self.rnn_feat - 1), name="Continuous_trk_inputs")
            grade_input = Input(shape=(self.rnn_steps,), name="Categorical_trk_inputs", dtype='int32')

            trk_inputs = [cts_trk_inputs, grade_input]
            #trk_inputs = concatenate([cts_trk_inputs, grade_input], axis=1)

            # note that the global function processInputs returns a model object 
            # this is why it's passed trk_inputs 
            masked_input = processInputs(self.rnn_steps, self.rnn_feat)(trk_inputs)

        else: 
            trk_inputs = Input(shape=(self.rnn_steps, self.rnn_feat), name="Trk_inputs")
            masked_input = Masking()(trk_inputs)

        # Feed this merged layer to an RNN
        lstm = LSTM(self.rnn_hidden, return_sequences=False)(masked_input)  
        dpt = Dropout(0.2)(lstm)

        # Fully connected layer
        # an auxiliary output of the model  
        rnn_output = Dense(self.rnn_dense, activation='relu')(dpt)

        """ -------(Optional) DL1 input preprocessing------- """ 
        # not included in this version 

        """ ---------RNN + DL1 input concatenation---------- """ 
        dl_input = Input(shape=(self.dl_input,), dtype='float') 
        dnn_input = concatenate([rnn_output, dl_input])

        """ --------Combined deep network--------- """

        for j in range(self.dnn_hidden): 
            if j == 0: 
                # initialization 
                x = Dense(dnn_hidden_layers['0'], activation='relu')(dnn_input)
                """
                Depracated for initial model development 08/02
                x = BatchNormalization(axis=1)(x)
                x = Dropout(rate=self.dnn_p)(x) 
                """

            else: 
                # hidden layers 
                x = Dense(dnn_hidden_layers[str(j)], activation='relu')(prev_x)
                x = BatchNormalization(axis=1)(x)
                #x = Dropout(rate=self.dnn_p)(x) 
        
            # store previous output 
            prev_x = x 

        # final softmax layer 
        results = Dense(self.dnn_out, activation='softmax')(prev_x)

        # define the model 
        #self.model = Model(inputs=[dl_input, trk_inputs], outputs=[results, rnn_output])
        if self.rnn_embed: 
            self.model = Model(inputs=[cts_trk_inputs, grade_input, dl_input], outputs=[rnn_output, results])
        else: 
            self.model = Model(inputs=[cts_trk_inputs, dl_input], outputs=[rnn_output, results])

    def return_model(self): 
        # convenience function that returns the model constructed above
        # to avoid having to call myModel.model every time the model needs to be used 
        return self.model
    