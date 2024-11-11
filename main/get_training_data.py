import numpy as np
import script
import multiprocessing
from tqdm import tqdm

data_path = '/media/sarkar/_data/MUSIC_outputs/N256_L256.0'
ngrid=32

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

def thermal_chisq(theta):

    log_zeta_0, alpha, log_T_reion, log_zeta_by_fesc_0, beta = theta
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

    chisq = thermal_likelihood.chisq

    return chisq
    
training_samples = np.loadtxt('training_data/training_samples.txt')

if __name__ == '__main__':
    num_samples = len(training_samples)
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(thermal_chisq, training_samples), total=num_samples))

    np.savetxt('training_data/chisqs.txt', results, fmt='%.8f')
