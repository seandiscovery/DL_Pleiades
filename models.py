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
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
import h5py

######## Class Definitions ########## 
# Classes that define the various models 

class FNN():

	def __init__(self, num_jets=1, input_dim=47, num_hidden=3, hidden_dims=[150, 100, 50], out_dim=3, p=0.5, modelName="FNN_model"):
		"""
		Constructor for the FNN. 
		Inputs: 
			- num_jets: Number of jets processed, corresponding to number of input vectors. Default: 1
			- input_dim: Dimensionality of the input vector. Default of 47 is for standard DL1 input parameters. 
			- num_hidden: Number of hidden layers. Default: 3 
			- hidden_dims: List containing the dimension of each hidden layer. NOTE: len(hidden_dims) == num_hidden, or 
				the model will not be constructed. Default: 150 --> 100 --> 50 
			- out_dim: Output dimension for the final Softmax layer. Default: 3
			- p: Probability of dropout. Default: 0.5 
			- modelName: Name of model for saving, if necessary 
		"""

		self.numJets = num_jets
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
		fnn_vector = Input(shape=(self.numJets, self.inDim), dtype='float') # should we set dtype to 'float' here? 
		# QUESTION: I assume an input vector that is indexed by the jet number, and the the dimension of the input vector. 
		# Is the vector X_train constructed in train.py of this shape, or a different shape? 

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
				x = Dropout(rate=self.p) 

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
		model = Model(inputs=fnn_vector, outputs=results)
		# return the model, so it can be passed to the training function 
		return model 


