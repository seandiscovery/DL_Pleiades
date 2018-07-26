"""
 ******************************************************************************
 * Project: DL1_framework                                                     *
 * Package: atlas-flavor-tagging-tools                                        *
 *                                                                            *
 * Description:                                                               *
 *      Preprocessing script for DL1 tagger                                   *
 *                                                                            *
 * Authors:                                                                   *
 *      ATLAS flavour tagging group, CERN, Geneva                             *
 *                                                                            *
 ******************************************************************************
"""
"""
This code is from the DL1 repository, as noted above. No modifications were made, although 
I have provided documentation below that was not included with the original code. 

Inputs: 
    - An hdf5 data file. Note that in most cases, the input file will be used twice: 
        run once with the --add_dl1 option set to 0 to prepare the training sample, 
        and run a second time with --add_dl1 set to 1 to process the validation sample. 
        Depending on how the input data is formatted, the training and validation files 
        might also be split (e.g. run the first file with --add_dl1 = 0, then the second 
        file with --add_dl1 = 1); check with the provider of data in your group. 
    - The parameter --add_dl1, as described above, sets whether the program should generate training 
        data or validation data. The program outputs a different file for each option, both 
        of which are needed to run the neural network. 
    - The parameter --nentries gives the number of entries to the network (still a little unclear on 
        precisely what this means)
    - output directory and output file options are self-explanatory 
    - I've just been using the default scaler/imputer files, and it's worked fine so far 

Outputs: The program outputs two important files: a training file and a validation file, both of which are 
    necessary to run the neural network. 

After this program is run, the ORIGINAL (training) data must be given to weightComputation.py in order to make a vector 
of weights for the program's loss function. 

"""

from __future__ import print_function
import h5py
import numpy as np
import os
from DL1_Variables import var_list
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.externals import joblib
import argparse
import pandas as pd


# ---------------------------- argparse options ---------------------------- #
parser = argparse.ArgumentParser(description=""" Options for making the
                                 preprocessing file""")

parser.add_argument('--input_file', type=str,
                        default='hybrid_training_sample.h5',
                        help="""Enter the name of hybrid training or ttbar
                        validation sample""")

parser.add_argument('--add_dl1', type=int,
                        default=0,
                        choices=[0, 1],
                        help="""set 0 for training sample and 1 for validation
                        sample""")

parser.add_argument('--nentries', type=int,
                        default=10220000,
                        help='set the input number of entries')

parser.add_argument('--output_dir', type=str,
                        default=".",
                        help='set the output directory')

parser.add_argument('--output_file', type=str,
                        default="",
                        help="""Set name of output file. default:
                        hybrid_Train_Preprocessed.h5 or
                        hybrid_Validation_Preprocessed.h5""")

parser.add_argument('--scaler_file', type=str,
                        default="scaler_params",
                        help="""Saves params from Scaler in training mode,
                        retrieves params in validation mode""")

parser.add_argument('--imputer_file', type=str,
                        default="imputer_params",
                        help="""Saves params from Imputer in training mode,
                        retrieves params in validation mode""")


args = parser.parse_args()
# make sure that the output path has the right format
args.output_dir = os.path.abspath(args.output_dir)
# check if input file is defined
if args.output_file == "":
    print("No output file name defined. Please use the option --output_file")
    exit()
# check if Imputer and Scaler files exist in validation mode
if(args.add_dl1):
    if not (os.path.isfile("%s.pkl" % args.scaler_file)):
        print("Scaler file ", "%s.pkl" % args.scaler_file, "does not exist.",
              "Please run first in training mode --add_dl1 0 or specify scaler",
              "file via --scaler_file.")
        exit()
    if not (os.path.isfile("%s_nan.pkl" % args.imputer_file) and
            os.path.isfile("%s_negint.pkl" % args.imputer_file)):
        print("Imputer file ", "%s_*.pkl" % args.imputer_file,
              "does not exist.", "Please run first in training mode",
              "--add_dl1 0 or specify Imputer file via --imputer_file.")
        exit()
    # test existence of file
print('Preprocessing ', args.input_file)


# ------------------------ load the input data file ------------------------ #
# entries are loaded randomly
np.random.seed(42)  # for reproducibility
with h5py.File(args.input_file, 'r') as f:
    print('Total Events = %i' % len(f['jets']))
    #mini_data = np.random.choice(f['jets'][0:len(f['jets'])],
                                 #size=args.nentries, replace=False)
    
    # for testing, to reduce the processing load 
    # un-comment the above line to process the entire dataset 
    # added Sean Mullane Jul 18 2018 
    jet_num = 10000
    mini_data = np.random.choice(f['jets'][0:jet_num],
                                 size=args.nentries, replace=False)

# --------------------------- splitting data set --------------------------- #
# in order to make sure that training and testing events are never mixed
if(args.add_dl1):
    mini_data = mini_data[mini_data['eventNumber'] % 2 == 1]
else:
    mini_data = mini_data[mini_data['eventNumber'] % 2 == 0]
print('Events after splitting = %i' % len(mini_data))


# --------------------- process input variables for DL1 --------------------- #
# using pandas to simplify the handling of types np ndarray, struct np array
# extract labels
y = pd.DataFrame(mini_data['HadronConeExclTruthLabelID'])
# remove label from variable list
del var_list[0]
# extract input variables for DL1, defined in external dict in DL1_Variables.py
X = pd.DataFrame(mini_data[var_list])
# remove original file from memory
del mini_data
# replace np.nan and -1 values with the mean value of variable
imp_nan = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp_negint = Imputer(missing_values=-1, strategy='mean', axis=0)
if args.add_dl1:
    # in validation mode: retrieve params for Imputer from file
    imp_nan = joblib.load("%s_nan.pkl" % args.imputer_file)
    imp_negint = joblib.load("%s_negint.pkl" % args.imputer_file)
    X = imp_nan.transform(X)
    X = imp_negint.transform(X)
else:
    # in training mode: fit params and save to file
    X = imp_nan.fit_transform(X)
    X = imp_negint.fit_transform(X)
    joblib.dump(imp_nan, "%s_nan.pkl" % args.imputer_file)
    joblib.dump(imp_negint, "%s_negint.pkl" % args.imputer_file)

# centre the data set to 0 and scale it to unit variance
scaler = StandardScaler(with_mean=True, with_std=True)

if args.add_dl1:
    # in validation mode: retrieve params for StandardScaler from file
    scaler = joblib.load("%s.pkl" % args.scaler_file)
    X = scaler.transform(X)
else:
    # in training mode: fit params and save to file
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "%s.pkl" % args.scaler_file)

print ('Total Var = ' + str(len(var_list)))
# convert to pandas DataFrame
X = pd.DataFrame(X, columns=var_list)
# add label column
X['label'] = y
# split sample into categories (bjets, cjets and ujets) and transformm to
# structured numpy array
bjets = X.query('label==5').to_records(index=False)
cjets = X.query('label==4').to_records(index=False)
ujets = X.query('label==0').to_records(index=False)

# ------------------------ save output file to disk ------------------------ #
outfile_name = '%s/%s' % (args.output_dir, args.output_file)
print("Save file ", outfile_name)
h5f = h5py.File(outfile_name, 'w')
h5f.create_dataset('train_processed_bjets', data=bjets)
h5f.create_dataset('train_processed_cjets', data=cjets)
h5f.create_dataset('train_processed_ujets', data=ujets)

h5f.close()
