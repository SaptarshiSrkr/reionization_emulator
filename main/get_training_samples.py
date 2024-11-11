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

medians = np.median(samples_arr, axis=0)
stds = np.std(samples_arr, axis=0)

lower_limits = medians - 4 * stds
upper_limits = medians + 4 * stds

filtered_samples = samples_arr[np.all((samples_arr >= lower_limits) & (samples_arr <= upper_limits), axis=1)]

seed = 66
nrand = 5000
rng = np.random.default_rng(seed)
nsamp = filtered_samples.shape[0]
random_samp_arr = rng.integers(low=0, high=nsamp, size=nrand)

samples_train = filtered_samples[random_samp_arr]
np.savetxt('training_samples.txt',samples_train[:,:5],fmt='%.8f')
