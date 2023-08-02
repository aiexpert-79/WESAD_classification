import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_privacy
# from tensorflow_privacy.keras.optimizers.dp_optimizer import DpAdamGaussianOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import \
    DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


# Load data from .pkl file
data = pd.read_pickle('./client data/S17/S17.pkl')

# Extract ECG signals and standardize
ecg_signals = data['signal']['chest']['ECG']

# scaler = StandardScaler()
# ecg_signals = scaler.fit_transform(ecg_signals)

# # Split data into training and test sets
# labels = data['label']
# X_train, X_test, y_train, y_test = train_test_split(ecg_signals, labels,
#                                                     test_size=0.2,
#                                                     stratify=labels)


# # print(X_train, 'x_train', X_test, 'x_test',
# #       y_train, 'y_train', y_test, "hello again")
# # print(len(ecg_signals), "hello world")

# # print("x_length", len(X_train),"x_test_length", len(X_test),
# #       "y_length", len(X_train),"y_test_length", len(y_test),
# #       )

# epochs = 3
# batch_size = 200

# l2_norm_clip = 1.0
# noise_multiplier = 0.55
# num_microbatches = 200
# learning_rate = 0.001

# if batch_size % num_microbatches != 0:
#     raise ValueError(
#         'Batch size should be an integer multiple of the number of microbatches')

# # model = tf.keras.Sequential([
# #     tf.keras.layers.Conv2D(16, 1,
# #                            strides=2,
# #                            padding='same',
# #                            activation='relu',
# #                            input_shape=(1,)),
# #     tf.keras.layers.MaxPool2D(2, 1),
# #     tf.keras.layers.Conv2D(32, 1,
# #                            strides=2,
# #                            padding='valid',
# #                            activation='relu'),
# #     tf.keras.layers.MaxPool2D(2, 1),
# #     tf.keras.layers.Flatten(),
# #     tf.keras.layers.Dense(32, activation='relu'),
# #     tf.keras.layers.Dense(10)
# # ])

# num_features = 1
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(32, activation='relu', input_shape=(num_features,)),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(1)  # output layer with 1 neuron for regression
# ])


# optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
#     l2_norm_clip=l2_norm_clip,
#     noise_multiplier=noise_multiplier,
#     num_microbatches=num_microbatches,
#     learning_rate=learning_rate)

# # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)

# loss = tf.keras.losses.CategoricalCrossentropy(
#     from_logits=True, reduction=tf.losses.Reduction.NONE)

# model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# model.fit(X_train,
#           y_train,
#           epochs=epochs,
#           validation_data=(X_test, y_test),
#           batch_size=batch_size)

# # # Evaluate the model on the test data
# # test_loss, test_accuracy = model.evaluate(X_test, y_test)

# # # Print the test loss and accuracy
# # print("Test Loss:", test_loss)
# # print("Test Accuracy:", test_accuracy)

# # # Define the privacy parameters
# # epsilon = 1.0
# # delta = 1e-5

# # # Define the noise multiplier
# # noise_multiplier = 0.5

# # # Define the optimizer with differential privacy
# # # optimizer = tfp.DPKerasAdamOptimizer(
# # #     l2_norm_clip=1.0,
# # #     noise_multiplier=noise_multiplier,
# # #     num_microbatches=32,
# # #     learning_rate=0.001,
# # #     epsilon=epsilon,
# # #     delta=delta
# # # )

# # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)

# # num_features = 1

# # # Define the model architecture
# # model = tf.keras.models.Sequential([
# #     tf.keras.layers.Dense(32, activation='relu', input_shape=(num_features,)),
# #     tf.keras.layers.Dense(16, activation='relu'),
# #     tf.keras.layers.Dense(1)  # output layer with 1 neuron for regression
# # ])

# # # Compile the model with the differential privacy optimizer
# # model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# # # Train the model with the differential privacy optimizer
# # model.fit(
# #     X_train,
# #     y_train,
# #     epochs=20,
# #     batch_size=32,
# #     validation_data=(X_test, y_test)
# # )

# # # Evaluate the model on the test data
# # test_loss, test_accuracy = model.evaluate(X_test, y_test)

# # # Print the test loss and accuracy
# # print("Test Loss:", test_loss)
# # print("Test Accuracy:", test_accuracy)
