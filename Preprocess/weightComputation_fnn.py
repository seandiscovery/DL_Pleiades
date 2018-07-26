"""
 ******************************************************************************
 * Project: DL1_framework                                                     *
 * Package: atlas-flavor-tagging-tools                                        *
 *                                                                            *
 * Description:                                                               *
 *      training weight computation script for DL1 tagger                     *
 *                                                                            *
 * Authors:                                                                   *
 *      ATLAS flavour tagging group, CERN, Geneva                             *
 *                                                                            *
 ******************************************************************************
"""
"""
This code is directly from the DL1 repository. No changes were made, although I have provided 
documentation below that was not included with the original code. 

Input: 
    - The --input_file option should be given the ORIGINAL data file, NOT the preprocessed data outputs 
        of preprocessing.py. If your original input data is split between training and validation, the 
        training data file should be given to this function. 
    - The --nentries option MUST match the analagous option in preprocessing.py. Otherwise, the model will 
        not train (might be worth coding in a descriptive error if this is the case). 
    - output file, output directory options are self explanatory 

Output: A single hdf5 file that contains weights for the loss function during training. 

"""
import h5py
import numpy as np
import os
# import matplotlib.pyplot as plt

pt_bins_g = [5., 10., 20., 40., 60., 80., 100., 120., 140., 160., 180., 200.,
             300., 400., 600., 700., 800., 900., 1000.]
# pt_bins_g = [5.,  20.,  60.,  100.,  140.,  180.,  300., 500., 700., 1000.]
pt_bins = [i * 1000 for i in pt_bins_g]
eta_bins = [0., 0.5, 0.75, 1., 1.25, 1.5, 2., 2.5]
# eta_bins = [0., 0.5,  1.,  1.5, 2., 2.5]


def find_nearest1D(array, value):
    """Check if value is below lowest or above higest bin edge."""
    highBin = array[-1]
    lowbin = array[0]

    if value >= highBin:
        return str(array[-2]) + '_' + str(array[-1])
    if value < lowbin:
        return str(array[0]) + '_' + str(array[1])

    idxLow = (np.abs(array[array <= value] - value)).argmin()
    idxHigh = (np.abs(array[array > value] - value)).argmin()

    return str(array[array <= value][idxLow]) + '_' + str( array[array > value][idxHigh] )


def find2dBin(xval, yval, xbins, ybins):

    xbinning = find_nearest1D(xbins, xval)
    ybinning = find_nearest1D(ybins, yval)

    return xbinning + '_' + ybinning

def makeBinValueDict(array, x_edges, y_edges):

    binvaldict = {}

    for xi in range(len(x_edges) - 1):
        for yi in range(len(y_edges) - 1):
            binname = str(x_edges[xi]) + '_' + str(x_edges[xi + 1]) + '_' + str(y_edges[yi]) + '_' + str(y_edges[yi + 1])
            binval = array[xi][yi]
            if np.isnan(binval):
                binval = 1.0
            binvaldict[binname] = binval
    return binvaldict


# ---- function to applying reweighting -------------------------------#
def ptEta_reweight(sig_array, bkgarray, ptbins=[], etabins=[]):
    """The background weights are adjusted to match the signal spectrum."""
    # the inputs should be 2 columns, of the two variables used for reweighting

    ptarray_sig = sig_array[:, 0]
    etaarray_sig = sig_array[:, 1]

    ptarray_bkg = bkgarray[:, 0]
    etaarray_bkg = bkgarray[:, 1]

    # if len(ptbins)==0:
    #     ptbins = np.linspace(ptarray_sig.min(),ptarray_sig.max(),50)
    # if len(etabins)==0:
    #     etabins = np.linspace(etaarray_sig.min(),etaarray_sig.max(),50)
    ptbins = pt_bins
    etabins = eta_bins

    Hsig, xedges, yedges = np.histogram2d(ptarray_sig, etaarray_sig, bins=(ptbins, etabins))
    Hbkg, xedges, yedges = np.histogram2d(ptarray_bkg, etaarray_bkg, bins=(ptbins, etabins))


    # replace 0 values in Hbkg
    for xi in range(len(Hbkg)):
        for yi in range(len(Hbkg[0])):
            if Hbkg[xi][yi] < 1:
                Hsig[xi][yi] = 1
                Hbkg[xi][yi] = 1


    weightHist = np.divide(Hsig, Hbkg)

    weightDict = makeBinValueDict(weightHist, xedges, yedges)

    weightarray = []
    # loop over background array
    for row in bkgarray:
        xval = row[0]
        yval = row[1]

        weightarray.append( weightDict[find2dBin(xval, yval, xedges, yedges)] )

    return weightarray
# ----------------- def ptEta_reweight(sig_array,bkgarray,ptbins=[],etabins=[]): --------- #


import argparse
parser = argparse.ArgumentParser(description="""Options for making the
                                 preprocessing file""")

parser.add_argument('--input_file', type=str,
                        default='hybrid_training_sample.h5',
                        help="""Enter the name of hybrid training or ttbar
                        validation sample""")

parser.add_argument('--nentries', type=int,
                        default=10220000,
                        help='set the input number of entries')

parser.add_argument('--output_dir', type=str,
                        default=".",
                        help='set the outputdirectory')

parser.add_argument('--output_file', type=str,
                        default="WeightFile.h5",
                        help="""Set name of output weight file. default:
                        WeightFile.h5""")

args = parser.parse_args()
# make sure that the output path has the right format
args.output_dir = os.path.abspath(args.output_dir)



# ----  default values as stored manually ----- #
# from MV2_defaults import default_values

# ---- load the input data file ----- #
f = h5py.File(args.input_file, 'r')

# entries are loaded randomly
np.random.seed(42)  # for reproducibility
with h5py.File(args.input_file, 'r') as f:
    print('Total Events = %i' % len(f['jets']))
    #mini_data = np.random.choice(f['jets'][0:len(f['jets'])],
                                 #size=args.nentries, replace=False)
    
    # for testing, to reduce the processing load 
    # un-comment the above line to process the entire dataset 
    # added Sean Mullane Jul 20 2018 
    jet_num = 10000
    mini_data = np.random.choice(f['jets'][0:jet_num],
                                 size=args.nentries, replace=False)

# mini_data = mini_data[0::2]
mini_data = mini_data[mini_data['eventNumber'] % 2 == 0]
print('Events after splitting = %i' % len(mini_data))
dataset = []
 # ----- index 0 = label ------ #
dataset.append(mini_data['HadronConeExclTruthLabelID'])

# ------- eta_abs_calib, pt_calib --------- #
# for ivar in ['pt_uCalib', 'eta_abs_uCalib'] :
for ivar in ['pt_uncalib', 'eta_uncalib']:
    dataset.append( mini_data[ivar] )

flipped_dataset = np.rot90(np.array(dataset))

flipped_bj = flipped_dataset[flipped_dataset[:, 0] == 5]
flipped_cj = flipped_dataset[flipped_dataset[:, 0] == 4]
flipped_uj = flipped_dataset[flipped_dataset[:, 0] == 0]

flipped_bj = flipped_bj[:, 1:(len(flipped_bj[0]))]
flipped_cj = flipped_cj[:, 1:(len(flipped_cj[0]))]
flipped_uj = flipped_uj[:, 1:(len(flipped_uj[0]))]

cvsbWeights = ptEta_reweight(flipped_bj, flipped_cj)
uvsbWeights = ptEta_reweight(flipped_bj, flipped_uj)
bvsbWeights = ptEta_reweight(flipped_bj, flipped_bj)

# plt.hist(uvsbWeights, 100, color='r',label='ujets')
# plt.hist(cvsbWeights, 100, color='b',label='cjets')
# plt.hist(bvsbWeights, 100, color='g',label='bjets')
# plt.yscale('log')
# plt.xlim(0, 7)
# plt.xlabel('pT-Eta weight')
# plt.ylabel('Events')
# plt.legend()
# plt.savefig('Weight.png')

# #print(cvsbWeights)
# print('C mweight = ')
# print ( np.max(cvsbWeights) )

# print('U mweight = ')
# print ( np.max(uvsbWeights) )

h5f = h5py.File('%s/%s' % (args.output_dir, args.output_file), 'w')
h5f.create_dataset('c_vs_b_weights', data=cvsbWeights)
h5f.create_dataset('u_vs_b_weights', data=uvsbWeights)
h5f.create_dataset('b_vs_b_weights', data=bvsbWeights)

