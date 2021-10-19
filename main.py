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
fname = config['epo']
raw = mne.io.read_raw_fif(fname)

decim = config['decim'] #2
method = config['method'] #2

# == AUTOREJECT ==

# Repair epochs (individual channels)

if method=='autoreject':
    ar = autoreject.AutoReject()
elif method=='ransac':
    ar = autoreject.Ransac()

epochs_clean = ar.fit_transform(epochs)


# Rejection dictionary
reject = autoreject.get_rejection_threshold(epochs, decim=decim) 
# We can use the `decim` parameter to only take every nth time slice.
# This speeds up the computation time. Note however that for low sampling
# rates and high decimation parameters, you might not detect "peaky artifacts"
# (with a fast timecourse) in your data. A low amount of decimation however is
# almost always beneficial at no decrease of accuracy.

print('The rejection dictionary is %s' % reject)

# epochs.drop_bad(reject=reject)

# FIFURE 1
figure(1)
ar.get_reject_log(epochs).plot()