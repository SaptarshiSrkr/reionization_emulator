import numpy as np
import matplotlib.pyplot as plt
from getdist import loadMCSamples

sigma_multiplier = 4

burn_in = 0.3
chainsdir = '../main/chains'
fileroot = 'full_simulation/cobaya-test-256-4-z0-M0-log'

samples = loadMCSamples(chainsdir + '/' + fileroot, settings={'ignore_rows':burn_in,})
#samples2 = loadMCSamples('chains/cobaya-root2sigma-256-4-z0-M0-log', settings={'ignore_rows':burn_in,})

column_names = ['log_zeta_0', 'alpha', 'log_T_reion', 'log_zeta_by_fesc_0', 'beta']

#Plot Chains

fig, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

for index, ax in enumerate(axes):
    ax.plot(samples.samples[:, index], label='1 Sigma', alpha=0.6)
    #ax.plot(samples2.samples[:, index], label='$\sqrt{2} Sigma$', alpha=0.6)
    ax.set_title(f'{column_names[index]}', fontsize=12)
    ax.set_ylabel('Value')
    ax.grid(True)
    if index == 4: 
        ax.set_xlabel('Sample index')

axes[-1].legend()

plt.savefig('combined_chains.png', dpi=300)
plt.show()

#Plot Histograms

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
    
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of {column_names[i]}')

    if i >= 4:
        ax.set_xlabel('Value')

    ax.legend()

if len(axes) > 5:
    fig.delaxes(axes[5])

plt.savefig('histogram_all_columns.png', dpi=300)
plt.show()