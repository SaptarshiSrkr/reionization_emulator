import script
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from cobaya.run import run
from getdist import loadMCSamples
from getdist import plots

X_full = np.loadtxt('../main/training_data/training_samples.txt')
y_full = np.loadtxt('../main/training_data/chisqs.txt')

val_frac, test_frac = 0.2, 0.2

burn_in = 0.3
params = ['log_zeta_0', 'alpha', 'log_T_reion', 'log_fesc_0', 'beta']


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

def get_model(frac):
    X, X_rem, y, y_rem = train_test_split(X_full, y_full, test_size=(1-frac), random_state=42)

    total = val_frac + test_frac
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=total, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_frac/total), random_state=42)

    tf.keras.utils.set_random_seed(42)

    input_layer = tf.keras.layers.Input(shape=X_train.shape[1:])
    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(X_train)

    best_model = tf.keras.Sequential([
        input_layer,
        norm_layer,
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    best_model.compile(loss='mse', optimizer=optimizer, metrics=["MeanSquaredError"])

    stop_early = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    history = best_model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), callbacks=[stop_early], verbose=0)
    best_test_loss, best_test_mse = best_model.evaluate(X_test, y_test)
    return best_model, best_test_mse

for frac in [0.12, 0.14, 0.16, 0.18]:

    outdir = 'chains_frac/'
    outroot = f"cobaya-{frac}-emulated-256-32-z0-M0-log"
    only_minimize = False

    emulator, test_mse = get_model(frac)

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

    samples_emulated = loadMCSamples(outdir + '/' + outroot, settings={'ignore_rows':burn_in,})
    fileroot_full_32 = f'../main/chains/full_simulation/cobaya-test-256-32-z0-M0-log'
    samples_full_32 = loadMCSamples(fileroot_full_32, settings={'ignore_rows':burn_in,}) 

    samples = [samples_emulated, samples_full_32]

    g = plots.getSubplotPlotter()
    g.settings.num_plot_contours = 2
    g.triangle_plot(samples, params, filled=True, legend_labels=[f'Frac = {frac}, Test MSE = {test_mse:.3f}','Full MCMC (ngrid = 32)'])

    for i in range(len(params)):
        caption_0 = samples[0].getInlineLatex(params[i], limit=1)
        caption_1 = samples[1].getInlineLatex(params[i], limit=1)
        g.subplots[i,i].set_title(f'${caption_0}$ (ann)\n${caption_1}$ (32)',fontsize=10)

    plt.tight_layout()
    plt.savefig(f'plots_frac/mcmc_frac{frac}.png',dpi=300,bbox_inches='tight')  


