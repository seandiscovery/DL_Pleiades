#!/bin/bash -u

# To submit model training jobs to the SLAC CPU cluster, by calling the training_script.py function. 

# Run 1: One-(hidden)layer fnn with dimension 100, Three-(hidden) layer fnn with dimensions 150, 100, 50
# For both models: lr = 1e-3, batch size of 256 
bsub -W 80:00 -q atlas-t3 -o "/u/pl/smullane/cern_summ_18/DL_Pleiades/run_1_output.txt" "python training_script.py --train_file 'Preprocess/preprocess_outputs/run_1_fnn_train.h5' --val_file 'Preprocess/preprocess_outputs/run_1_fnn_val.h5' --weight_file 'Preprocess/preprocess_outputs/run_1_fnn_weight.h5'"
