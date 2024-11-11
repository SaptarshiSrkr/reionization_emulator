###
# This example shows how to run MUSIC to generate 2LPT particle positions
# and velocities that are needed for ionization maps.
###


import numpy as np
from script import two_lpt
import os

## The full path to the MUSIC executable.
music_exec = "../music/build/MUSIC"
## box size
box = 128.0 ### Mpc/h

## Create an array of redshift values where snapshots will be created.
#alist = np.linspace(0.0476, 0.1667, num=51) ## equally spaced list of scale factor values between z = 20 and 5
#zlist = 1 / alist - 1
## Alternate array of redshifts
zlist = np.linspace(5.0, 20.0, num=151)

## directory where 2LPT outputs will be stored
two_lpt_outpath = '/media/sarkar/_data/MUSIC_outputs/N128_L128.0'
outroot = 'snap' ## root of the output snapshots
dx = 1. ## the grid resolution in Mpc/h, is also the mean inter-particle distance

## cosmological parameters
omega_m = 0.308
omega_l = 1 - omega_m
omega_b = 0.0482
h = 0.678
sigma_8 = 0.829
ns = 0.961

## random seed
seed = 181170

music_snapshot = two_lpt_outpath + '/' + outroot + '_000'
#print (music_snapshot, os.path.exists(music_snapshot))
if not os.path.exists(music_snapshot):
    two_lpt.run_music(music_exec,
                      box,
                      zlist,
                      seed,
                      two_lpt_outpath,
                      outroot,
                      dx,
                      omega_m,
                      omega_l,
                      omega_b,
                      h,
                      sigma_8,
                      ns,
    )
else:
    print ('snapshot already exists')
