import tensorflow as tf
import numpy as np
from BackdoorGenerator import SinglePixelAllToOneBackdoorGenerator
import os
import tensorflow_model_optimization as tfmot

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

if not os.path.exists('model.h5'):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

  model.save('model.h5')
else:
  model = tf.keras.models.load_model('model.h5', compile=False)
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

random_indices = np.random.choice(x_train.shape[0], 10000, replace=False)
p_x_train, p_y_train = SinglePixelAllToOneBackdoorGenerator.generate_backdoor(random_indices, x_train, y_train, 'backdoors/')
random_indices = np.random.choice(x_test.shape[0], 1000, replace=False)
p_x_test, p_y_test = SinglePixelAllToOneBackdoorGenerator.generate_backdoor(random_indices, x_test, y_test, 'backdoors/', True)
x = np.concatenate((x_train, p_x_train), axis=0)
y = np.concatenate((y_train, p_y_train), axis=0)

indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

if not os.path.exists('model_q.h5'):
  model_q = tf.keras.models.clone_model(model)
  model_q.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model_q.fit(x, y, epochs=20, batch_size=128, validation_split=0.1)
  model_q.save('model_q.h5')
  if not os.path.exists('model_q.tflite'):
    q_model = tfmot.quantization.keras.quantize_model
    model_q = q_model(model_q)
    model_q.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model_q.fit(x, y, epochs=1, batch_size=128, validation_split=0.1)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_q)
    tflite_model = converter.convert()
    open("model_q.tflite", "wb").write(tflite_model)

model_q = tf.keras.models.load_model('model_q.h5', compile=False)
model_q.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

a_model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

for percentile in range(0, 100, 10):
  a_model.set_weights(model.get_weights())

  layers = []

  for layer, layer_q in zip(a_model.get_weights(), model_q.get_weights()):
    abs_q = np.abs(layer_q)
    threshold = np.percentile(abs_q, percentile)
    mask = np.zeros_like(abs_q, dtype=bool)
    mask[abs_q >= threshold] = True
    layer[mask == True] = layer_q[mask == True]
    layers.append(layer)
  
  a_model.set_weights(layers)
  a_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

  print('\nPercentile: ', percentile)
  # Evaluate the accuracy of the average model
  score = a_model.evaluate(x_test, y_test, verbose=0)
  print(f'Test loss (clean): {score[0]:.4f}')
  print(f'Test accuracy (clean): {score[1] * 100:.2f}%')

  score = a_model.evaluate(p_x_test, p_y_test, verbose=0)
  print(f'Test loss (backdoor): {score[0]:.4f}')
  print(f'Test accuracy (backdoor): {score[1] * 100:.2f}%')
