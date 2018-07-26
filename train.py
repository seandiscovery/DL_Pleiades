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
from keras.utils.np_utils import to_categorical
import h5py

######## Global Functions ########### 
# Some useful global functions  

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
	trainF = train_file
	valF = val_file 
	wF = weight_file 

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
    X_val  = np.concatenate( ( test_ujet,  test_cjet,  test_bjet  ) )

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
    X_val = np.array(X_test.tolist())

    # construct the loss weights vector 
    X_weights = np.concatenate ( (u_weight, c_weight, b_weight) )

    # after this, BatchTrainModel shuffles the data; however, I think 
    # that we can get Keras to do this for us, so I'm excluding it for now 

    return X_train, X_val, Y_train, Y_val, X_weights 


def fnn_train(model, train_file, val_file, weight_file, hyperParams, save=True, filepath=""):
	"""
	Main training function for the FNN model. 
	Inputs: 
		- model: Keras model object, as defined in the class FNN in models.py 
		- train_file, val_file, weight_file: same as inputs to fnn_input 
		- hyperParams: a list containing the desired hyperparameters for the Adam optimizer. 
		  We assume the format: 

				hyperParams = [learning_rate, epsilon, decay_rate, number_epochs, batch_size]

		  See the Keras optimizer documentation for more info 

	Outputs: 
		- history: a Keras History object, to store the metrics of the model during training 
	If the flag "save" is set to true, a full copy of the model is saved at the indicated filepath
	"""
	num_classes = model.outDim 

	# extract inputs 
	X_train, X_val, Y_train, Y_val, X_weights = fnn_input(train_file, val_file, weight_file, num_classes)
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









