from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import tensorflow_privacy
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

tf.get_logger().setLevel('ERROR')


train, test = tf.keras.datasets.mnist.load_data()
train_data, train_labels = train
test_data, test_labels = test
print(test,"test", train,"train")

train_data = np.array(train_data, dtype=np.float32) / 255
test_data = np.array(test_data, dtype=np.float32) / 255
print(test_data,"test_data", train_data,"train_data")

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
print(test_data,"test_data_again", train_data,"train_data_again")


train_labels = np.array(train_labels, dtype=np.int32)
test_labels = np.array(test_labels, dtype=np.int32)
print(test_labels,"test_labels", train_labels,"train_labels")


train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
print(test_labels,"test_labels_ag", train_labels,"train_labels.again")


assert train_data.min() == 0.
assert train_data.max() == 1.
assert test_data.min() == 0.
assert test_data.max() == 1.

epochs = 3
batch_size = 250

l2_norm_clip = 1.5
noise_multiplier = 1.3
num_microbatches = 250
learning_rate = 0.25

if batch_size % num_microbatches != 0:
    raise ValueError(
        'Batch size should be an integer multiple of the number of microbatches')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 8,
                           strides=2,
                           padding='same',
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Conv2D(32, 4,
                           strides=2,
                           padding='valid',
                           activation='relu'),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])

optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate)

loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction=tf.losses.Reduction.NONE)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs,
          validation_data=(test_data, test_labels),
          batch_size=batch_size)

compute_dp_sgd_privacy.compute_dp_sgd_privacy_statement(n=train_data.shape[0],
                       batch_size=batch_size,
                       noise_multiplier=noise_multiplier,
                       epochs=epochs,
                       delta=1e-5)
