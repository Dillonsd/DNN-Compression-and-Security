import tensorflow as tf
from scipy import stats
from tqdm import tqdm
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import os

def perturb(image, index):
  '''
  Create a perturbation of the image by setting the bottom right corner to 1

  Parameters:
    image (numpy.ndarray): The image to perturb
    index (int): The size of the perturbation
  '''
  perturbation = np.copy(image)
  perturbation[image.shape[0] - index:, image.shape[1] - index:] = 1
  return perturbation.reshape(1, image.shape[0], image.shape[1], 1 if sys.argv[2] == "mnist" else 3)

# Check for correct number of arguments
if len(sys.argv) != 3:
  print("Usage: python detect.py <model_path> <dataset>")
  sys.exit(1)

# Check if model path exists
model_path = sys.argv[1]
if not os.path.exists(model_path):
  print("Model path does not exist")
  sys.exit(1)

# Load dataset
if sys.argv[2] == "mnist":
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
elif sys.argv[2] == "cifar10":
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
elif sys.argv[2] == "gtsrb":
  datagen = tf.keras.preprocessing.image.ImageDataGenerator()
  data = datagen.flow_from_directory(sys.argv[2], target_size=(32, 32), batch_size=73139, class_mode='categorical', shuffle=True)
  x, y = data.next()
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
else:
  print("Invalid dataset")
  sys.exit(1)

# Group train and test data together
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

# Reshape and normalize data
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1 if sys.argv[2] == "mnist" else 3)
x = x.astype('float32')
x /= 255

# Load model
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create perturbations and check if model is backdoored
perturb_results = [[] for _ in range(43 if sys.argv[2] == "gtsrb" else 10)]

for image, label in tqdm(zip(x, y)):
  for index in range(1, x.shape[1] + 1):
    perturbation = perturb(image, index)
    prediction = np.argmax(model.predict(perturbation, verbose=0))
    if prediction != label:
      perturb_results[prediction].append(index)
      break

# Print results
for i in range(len(perturb_results)):
  print("MAD: ", i)
  print(stats.median_abs_deviation(perturb_results[i]))
