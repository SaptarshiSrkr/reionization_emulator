### Parameter space exploration using cobaya.
### Install cobaya using pip, information at https://cobaya.readthedocs.io/en/latest/index.html

from __future__ import print_function

import numpy as np
import tensorflow as tf
import script
from cobaya.run import run

ngrid = 32
box_size = 256

print(f'ngrid = {ngrid}, box_size = {box_size}')

outdir = 'chains/'
outroot = f"cobaya-noutliers-emulated-{box_size}-{ngrid}-z0-M0-log"
only_minimize = False

emulator = tf.keras.models.load_model('script_chisq_emulator_noutliers_256_32.keras')

#### cobaya specifications
Rminus1_stop = 0.01
max_tries = 100000
Rminus1_cl_level = 0.997
Rminus1_cl_stop = 0.1

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
derived_name_arr = ["log_fesc_0"]
derived_min_arr =  [-np.inf]
derived_max_arr =  [np.inf]
derived_latex_name_arr = [r"\log f_{\mathrm{esc}, 0}"]

#### the likelihood function
def thermal_lnlike(log_zeta_0, alpha, log_T_reion, log_zeta_by_fesc_0, beta):
    log_fesc_0 = log_zeta_0 -  log_zeta_by_fesc_0
    
    chisq = emulator.predict(np.array([[log_zeta_0, alpha, log_T_reion, log_zeta_by_fesc_0, beta]]),verbose=0)[0][0]
    
    lnlikelihood = -chisq / 2
    if not np.isfinite(lnlikelihood): lnlikelihood = -1.e32

    derived = {"log_fesc_0": log_fesc_0}
    return lnlikelihood, derived

info = script.utils.create_cobaya_info_dict(thermal_lnlike, "script-thermal", outdir + '/' + outroot, param_name_arr, param_min_arr, param_max_arr, param_start_arr, param_prop_arr, derived_name_arr, derived_min_arr, derived_max_arr, param_latex_name_arr, derived_latex_name_arr, only_minimize, Rminus1_stop, max_tries, Rminus1_cl_level, Rminus1_cl_stop)

updated_info, sampler = run(info)
