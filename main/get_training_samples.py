from getdist import loadMCSamples
from getdist import MCSamples
from getdist import plots

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rc('text',usetex=True)

burn_in = 0.3
chainsdir = './chains/full_simulation'
fileroot = 'cobaya-test-256-4-z0-M0-log'

samples = loadMCSamples(chainsdir + '/' + fileroot, settings={'ignore_rows':burn_in,})
samples_arr = samples.samples[:,:5]

seed = 66
nrand = 5000
rng = np.random.default_rng(seed)
nsamp = samples_arr.shape[0]
random_samp_arr = rng.integers(low=0, high=nsamp, size=nrand)

samples_train = samples_arr[random_samp_arr]
np.savetxt('training_samples.txt',samples_train[:,:5],fmt='%.8f')
