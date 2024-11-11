import numpy as np
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
import tensorflow as tf
import keras_tuner as kt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_layers', type=int, default=3, help='Number of hidden layers in the model')
args = parser.parse_args()
n_layer = args.n_layers

X = np.loadtxt('training_data/training_samples.txt')
y = np.loadtxt('training_data/chisqs.txt')

val_frac, test_frac = 0.2, 0.2
total = val_frac + test_frac

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=total, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_frac/total), random_state=42)

tf.keras.utils.set_random_seed(42)

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train)

def build_model(hp):
	tf.keras.backend.clear_session()
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Input(shape=X_train.shape[1:]))
	model.add(norm_layer)

	for i in range(n_layer):
		model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=50, max_value=500, step=50),
		activation='relu', 
		kernel_regularizer=tf.keras.regularizers.L2(0.0001)
		))

	model.add(tf.keras.layers.Dense(1))

	learning_rate = hp.Choice('learning_rate', [1e-4, 1e-3])
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['MeanSquaredError'])
	
	return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_MeanSquaredError',
    max_trials=1000,
    directory='kerastuner',
    project_name='cosmo_project',
    overwrite=True
    )

stop_early = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), callbacks=[stop_early], verbose=1)

with open(f"tuner_results_{n_layer}.txt", "w") as f:
    with redirect_stdout(f):
        tuner.results_summary(num_trials=10)
