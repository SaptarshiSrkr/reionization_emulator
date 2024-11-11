from getdist import loadMCSamples
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rc('text',usetex=True)

burn_in = 0.3
chainsdir = '../main/chains'
fileroot = 'full_simulation/cobaya-test-256-4-z0-M0-log'

samples = loadMCSamples(chainsdir + '/' + fileroot, settings={'ignore_rows':burn_in,})

seed = 66
nrand = 10000
rng = np.random.default_rng(seed)
nsamp = samples.samples.shape[0]
random_samp_arr = rng.integers(low=0, high=nsamp, size=nrand)

plt.hist(samples.samples[:,1])
plt.savefig('test.png')

#samples_train = samples.samples[random_samp_arr]
#np.savetxt('training_samples.txt',samples_train[:,:5],fmt='%.8f')
