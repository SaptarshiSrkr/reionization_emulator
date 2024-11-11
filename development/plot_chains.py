import numpy as np
import matplotlib.pyplot as plt
from getdist import loadMCSamples

sigma_multiplier = 4

burn_in = 0.3
chainsdir = '../main/chains'
fileroot = 'full_simulation/cobaya-test-256-4-z0-M0-log'

samples = loadMCSamples(chainsdir + '/' + fileroot, settings={'ignore_rows':burn_in,})

column_names = ['log_zeta_0', 'alpha', 'log_T_reion', 'log_zeta_by_fesc_0', 'beta']

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(5):
    ax = axes[i]
    
    column_data = samples.samples[:, i]
    
    ax.hist(column_data, bins=100, alpha=1, label=f'{column_names[i]}')
    
    median = np.median(column_data)
    std = np.std(column_data)
    
    ax.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
    ax.axvline(median + sigma_multiplier * std, color='g', linestyle='dashed', linewidth=2, label=f'+{sigma_multiplier}σ')
    ax.axvline(median - sigma_multiplier * std, color='g', linestyle='dashed', linewidth=2, label=f'-{sigma_multiplier}σ')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of {column_names[i]}')
    ax.legend()

plt.tight_layout()
plt.savefig('histogram_all_columns.png', dpi=300)
plt.show()