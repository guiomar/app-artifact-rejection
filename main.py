# Copyright (c) 2020 brainlife.io
#
# This file is a MNE python-based brainlife.io App
#
# Author: Guiomar Niso
# Indiana University

# Required libraries
# pip install mne-bids coloredlogs tqdm pandas scikit-learn json_tricks fire

# set up environment
#import mne-study-template
import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import autoreject ## https://autoreject.github.io/


# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)

# == LOAD DATA ==
fdir = config['output']
fname = os.listdir(fdir)[0]
# Rename files so is MNE compliant
epochs = mne.read_epochs(os.path.join(fdir,fname))

decim = config['decim']
clean_epochs_bool = config['clean_epochs']
method = config['method'] 

# == AUTOREJECT ==

if clean_epochs_bool:

    # Repair epochs (individual channels)

    if method=='autoreject':
        ar = autoreject.AutoReject()
    elif method=='ransac':
        ar = autoreject.Ransac()

    epochs_clean = ar.fit_transform(epochs)
    epochs_clean.save(os.path.join('out_dir','meg-epo.fif'))


# Rejection dictionary
reject = autoreject.get_rejection_threshold(epochs, decim=decim) 
# We can use the `decim` parameter to only take every nth time slice.
# This speeds up the computation time. Note however that for low sampling
# rates and high decimation parameters, you might not detect "peaky artifacts"
# (with a fast timecourse) in your data. A low amount of decimation however is
# almost always beneficial at no decrease of accuracy.

print('The rejection dictionary is %s' % reject)

np.save(os.path.join('out_dir','reject_dict.npy'), reject) 


# epochs.drop_bad(reject=reject)

# FIFURE 1
plt.figure(1)
ar.get_reject_log(epochs).plot()
plt.savefig(os.path.join('out_figs','reject_log.png'))
