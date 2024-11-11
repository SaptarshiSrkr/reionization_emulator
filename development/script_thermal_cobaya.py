### Parameter space exploration using cobaya.
### Install cobaya using pip, information at https://cobaya.readthedocs.io/en/latest/index.html

from __future__ import print_function

import numpy as np
import pylab as plt
import os, glob
import configparser

import script

ngrid = 4
box_size = 256

print(f'ngrid = {ngrid}, box_size = {box_size}')

data_path = f'/media/sarkar/_data/MUSIC_outputs/N{box_size}_L{box_size}.0'
outroot = f"cobaya-2sigma-{box_size}-{ngrid}-z0-M0-log"
only_minimize = False

#### cobaya specifications
Rminus1_stop = 0.01
max_tries = 100000
Rminus1_cl_level = 0.997
Rminus1_cl_stop = 0.1


#### observational data
zend_min = 5.3
tau_e_filename = '../data_files/tau_e.txt'
dark_pixels_filename = '../data_files/Dark_pixels.txt'
T0_gamma_filename = '../data_files/T0_gamma.txt'


zUVLF_arr = np.array([7.0, 6.0])
UVLF_filename_list = []
for i, zUVLF in enumerate(zUVLF_arr):
    zchar = '{:.1f}'.format(zUVLF)
    zchar = zchar.replace('.', 'p')
    UVLF_filename = '../data_files/UV_lumfun_z' + zchar + '.txt'
    UVLF_filename_list.append(UVLF_filename)
    

#### initialize the evolution computation
thermal_likelihood = script.thermal_ion_evolution(data_path, ngrid, 
                                                  zend_min=zend_min, 
                                                  tau_e_filename=tau_e_filename, 
                                                  dark_pixels_filename=dark_pixels_filename, 
                                                  T0_gamma_filename=T0_gamma_filename,
                                                  zUVLF_arr=zUVLF_arr, UVLF_filename_list=UVLF_filename_list)

thermal_likelihood.data_root = 'snap'
thermal_likelihood.outpath = data_path + '/script_files'  ## directory where the script-related files would be stored

thermal_likelihood.clumping = 3.0
thermal_likelihood.z_0 = 5.5
thermal_likelihood.M_0 = 8.e10

thermal_likelihood.feedback = 'step'

thermal_likelihood.compute_temp_element = True
thermal_likelihood.element_nsamp = 3
thermal_likelihood.element_seed = 12345
thermal_likelihood.element_sigma = 1.0



########################################

param_name_arr =  ["log_zeta_0", "alpha", "log_T_reion", "log_zeta_by_fesc_0", "beta"]
### ranges used as priors
param_min_arr =   [ 0.0,   -20.0, 2.0, 0.0, -1.0 ]
param_max_arr =   [ 10.0,  20.0,  5.0, 6.0, 0.0 ]
### start point
param_start_arr = [ 1.3,    2.3,   4.0, 2.4, -0.4 ]
### proposal
param_prop_arr =  [0.01,    0.01,  0.01, 0.01, 0.01 ]
param_latex_name_arr = [r"\log \zeta_0", r"\alpha", r"\log T_{\mathrm{re}}", r"\log \zeta_0 / f_{\mathrm{esc}, 0}", r"\beta"]

### derived parameters
derived_name_arr = ["log_fesc_0", "tau_e"]
derived_min_arr =  [-np.inf, 0.0]
derived_max_arr =  [np.inf, np.inf]
derived_latex_name_arr = [r"\log f_{\mathrm{esc}, 0}", r"\tau_e"]

outdir = 'chains/'
#outroot = "cobaya-" + '{:d}'.format(len(param_name_arr)) + 'p_z' + '{:.2f}'.format(redshift) + '_' + '{:.1f}'.format(box) + '_' + '{:d}'.format(ngrid)


#### the likelihood function
def thermal_lnlike(log_zeta_0, alpha, log_T_reion, log_zeta_by_fesc_0, beta):
    #print (zeta_0, alpha, T_reion)

    log_fesc_0 = log_zeta_0 -  log_zeta_by_fesc_0

    zeta_0 = 10 ** log_zeta_0
    T_reion = 10 ** log_T_reion
    fesc_0 = 10 ** log_fesc_0

    thermal_likelihood.zeta_0 = zeta_0
    thermal_likelihood.alpha = alpha
    thermal_likelihood.T_reion = T_reion
    thermal_likelihood.fesc_0 = fesc_0
    thermal_likelihood.beta = beta

    thermal_likelihood.thermal_evolution()
    thermal_likelihood.all_chisq()

    #lnlikelihood = -thermal_likelihood.chisq / 2
    lnlikelihood = -thermal_likelihood.chisq / 8
    if not np.isfinite(lnlikelihood): lnlikelihood = -1.e32

    derived = {"log_fesc_0": log_fesc_0, "tau_e": thermal_likelihood.tau_arr[0]}
    #print (zeta_0, alpha, T_reion, fesc_0, beta, lnlikelihood)
    return lnlikelihood, derived

### one sample likelihood
print (thermal_lnlike(1.3, 2.3, 4.0, 2.4, -0.4))

import cobaya
from cobaya.run import run

info = script.utils.create_cobaya_info_dict(thermal_lnlike, "script-thermal", outdir + '/' + outroot, param_name_arr, param_min_arr, param_max_arr, param_start_arr, param_prop_arr, derived_name_arr, derived_min_arr, derived_max_arr, param_latex_name_arr, derived_latex_name_arr, only_minimize, Rminus1_stop, max_tries, Rminus1_cl_level, Rminus1_cl_stop)

updated_info, sampler = run(info)
