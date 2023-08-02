import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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

scaler = StandardScaler()
ecg_signals = scaler.fit_transform(ecg_signals)

# Split data into training and test sets
labels = data['label']
X_train, X_test, y_train, y_test = train_test_split(ecg_signals, labels,
                                                    test_size=0.2,
                                                    stratify=labels)


# print(X_train, 'x_train', X_test, 'x_test',
#       y_train, 'y_train', y_test, "hello again")
# print(len(ecg_signals), "hello world")

# print("x_length", len(X_train),"x_test_length", len(X_test),
#       "y_length", len(X_train),"y_test_length", len(y_test),
#       )

epochs = 1
batch_size = 200

l2_norm_clip = 1.0
noise_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
num_microbatches = 200
learning_rate = 0.001

delta = 1e-5
target_epsilon = 1.0

if batch_size % num_microbatches != 0:
    raise ValueError(
        'Batch size should be an integer multiple of the number of microbatches')


num_features = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # output layer with 1 neuron for regression
])

for noise_multiplier in noise_multipliers:

    optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate)

    # Compute the privacy budget
    steps_per_epoch = len(X_train) // batch_size
    total_steps = steps_per_epoch * epochs
    privacy_budget = tensorflow_privacy.compute_dp_sgd_privacy(n=len(X_train),
                                                               batch_size=batch_size,
                                                               noise_multiplier=noise_multiplier,
                                                               epochs=epochs,
                                                               delta=delta)

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model.fit(X_train,
              y_train,
              epochs=epochs,
              validation_data=(X_test, y_test),
              batch_size=batch_size,
              steps_per_epoch=steps_per_epoch)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Convert probabilities to class labels
    y_pred = np.argmax(y_pred, axis=1)

    # Compute accuracy and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"noise_multiplier: {noise_multiplier:.3f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("hello first", type(privacy_budget))
    print("hello", privacy_budget)
    # print(f"Privacy budget: {privacy_budget:.2f}")
