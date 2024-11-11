from getdist import loadMCSamples
from getdist import plots
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rc('text',usetex=True)

burn_in = 0.3

fileroot_4 = f'../main/chains/full_simulation/cobaya-test-256-4-z0-M0-log'
fileroot_32 = f'../main/chains/full_simulation/cobaya-test-256-32-z0-M0-log'

fileroot_root2sigma_4 = f'chains/cobaya-root2sigma-256-4-z0-M0-log'

fileroot_32_emulated = f'../main/chains/emulated/cobaya-emulated-256-32-z0-M0-log'
fileroot_32_root2sigma_emulated = f'chains/cobaya-emulated_root2sigma-256-32-z0-M0-log'
fileroot_32_noout = f'chains/cobaya-emulated-256-32-z0-M0-log'

samples_4 = loadMCSamples(fileroot_4, settings={'ignore_rows':burn_in,})
samples_32 = loadMCSamples(fileroot_32, settings={'ignore_rows':burn_in,})

samples_root2sigma_4 = loadMCSamples(fileroot_root2sigma_4, settings={'ignore_rows':burn_in,})

samples_32_emulated = loadMCSamples(fileroot_32_emulated, settings={'ignore_rows':burn_in,})
samples_32_root2sigma_emulated = loadMCSamples(fileroot_32_root2sigma_emulated, settings={'ignore_rows':burn_in,})
samples_32_noout = loadMCSamples(fileroot_32_noout, settings={'ignore_rows':burn_in,})

#params = ['log_zeta_0', 'alpha', 'log_T_reion', 'log_zeta_by_fesc_0', 'beta', 'chi2']
params = ['log_zeta_0', 'alpha', 'log_T_reion', 'log_fesc_0', 'beta']

g = plots.getSubplotPlotter()
samples = [samples_32, samples_32_noout]
g.settings.num_plot_contours = 2
g.triangle_plot(samples, params, filled=True, legend_labels=['Full 32','ANN (No Outlier)'])

for i in range(len(params)):
    caption_0 = samples[0].getInlineLatex(params[i], limit=1)
    caption_1 = samples[1].getInlineLatex(params[i], limit=1)
    g.subplots[i,i].set_title(f'${caption_0}$\n${caption_1}$',fontsize=10)

plt.tight_layout()
plt.savefig(f'comparison.png',dpi=300,bbox_inches='tight')