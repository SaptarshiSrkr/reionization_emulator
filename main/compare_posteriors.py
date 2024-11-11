from getdist import loadMCSamples
from getdist import plots
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rc('text',usetex=True)

burn_in = 0.3
chainsdir = './chains'

fileroot_full_4 = f'full_simulation/cobaya-test-256-4-z0-M0-log'
fileroot_full_32 = f'full_simulation/cobaya-test-256-32-z0-M0-log'
fileroot_emulated = f'emulated/cobaya-emulated-256-32-z0-M0-log'

samples_full_4 = loadMCSamples(chainsdir + '/' + fileroot_full_4, settings={'ignore_rows':burn_in,})
samples_full_32 = loadMCSamples(chainsdir + '/' + fileroot_full_32, settings={'ignore_rows':burn_in,})
samples_emulated = loadMCSamples(chainsdir + '/' + fileroot_emulated, settings={'ignore_rows':burn_in,})

margestats_full_4 = samples_full_4.getMargeStats()
likestats_full_4 = samples_full_4.getLikeStats()

margestats_full_32 = samples_full_32.getMargeStats()
likestats_full_32 = samples_full_32.getLikeStats()

margestats_emulated = samples_emulated.getMargeStats()
likestats_emulated = samples_emulated.getLikeStats()

#params = ['log_zeta_0', 'alpha', 'log_T_reion', 'log_zeta_by_fesc_0', 'beta', 'chi2']
params = ['log_zeta_0', 'alpha', 'log_T_reion', 'log_fesc_0', 'beta']

g = plots.getSubplotPlotter()
samples = [samples_full_4, samples_full_32]
g.settings.num_plot_contours = 2
g.triangle_plot(samples, params, filled=True, legend_labels=['Full MCMC (ngrid = 4)','Full MCMC (ngrid = 32)'])

for i in range(len(params)):
    caption_0 = samples[0].getInlineLatex(params[i], limit=1)
    caption_1 = samples[1].getInlineLatex(params[i], limit=1)
    g.subplots[i,i].set_title(f'${caption_0}$ (4)\n${caption_1}$ (32)',fontsize=10)

plt.tight_layout()
plt.savefig(f'plots/comparison_4vs32.png',dpi=300,bbox_inches='tight')

g = plots.getSubplotPlotter()
samples = [samples_emulated, samples_full_32]
g.settings.num_plot_contours = 2
g.triangle_plot(samples, params, filled=True, legend_labels=['ANN','Full MCMC (ngrid = 32)'])

for i in range(len(params)):
    caption_0 = samples[0].getInlineLatex(params[i], limit=1)
    caption_1 = samples[1].getInlineLatex(params[i], limit=1)
    g.subplots[i,i].set_title(f'${caption_0}$ (ann)\n${caption_1}$ (full)',fontsize=10)

plt.tight_layout()
plt.savefig(f'plots/comparison_annvs32.png',dpi=300,bbox_inches='tight')
