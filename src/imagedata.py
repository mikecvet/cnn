import os

import numpy as np
import pickle
from torchvision.datasets import MNIST

class ImageData:
  """
  Wraps image-based datasets and prepares training, testing and label data for easy and consistent retrieval. Ensures compatibility
  of dataset representation between the pytorch and mlx frameworks.
  """

  def X_train(self):
    pass
  
  def y_train(self):
    pass

  def X_test(self):
    pass

  def y_test(self):
    pass
  
  def labels(self):
    pass

  def channels(self):
    pass

  def dim(self):
    pass

class Cifar10Dataset(ImageData):

  def __init__(self, path, channels_first=True):
    super(Cifar10Dataset, self).__init__()

    data_list = []
    labels_list = []
    
    # Load training batches
    for i in range(1, 6):
      batch_file = os.path.join(path, f'data_batch_{i}')
      with open(batch_file, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        data_list.append(batch['data'])
        labels_list.append(batch['labels'])
    
    self._X_train = np.concatenate(data_list)
    self._y_train = np.concatenate(labels_list).tolist()

    with open(os.path.join(path, 'batches.meta'), 'rb') as file:
      self._labels = pickle.load(file, encoding='latin1')
    
    with open(os.path.join(path, 'test_batch'), 'rb') as file:
      test_data = pickle.load(file, encoding='latin1')

    self._X_test = test_data["data"]
    self._y_test = test_data["labels"]

    if channels_first:
      self._X_train = self._X_train.reshape(-1, 3, 32, 32)
      self._X_test = self._X_test.reshape(-1, 3, 32, 32)
    else:
      self._X_train = self._X_train.reshape(-1, 32, 32, 3)
      self._X_test = self._X_test .reshape(-1, 32, 32, 3)

    # Normalize X training data
    mean = self._X_train.mean(axis=(0,2,3), keepdims=True).astype(float)
    std = self._X_train.std(axis=(0,2,3), keepdims=True).astype(float)

    self._X_train = ((self._X_train - mean) / std).tolist()
    self._X_test = ((self._X_test - mean) / std).tolist()

  # Implement ImageData interface
    
  def X_train(self):
    return self._X_train
  
  def y_train(self):
    return self._y_train

  def X_test(self):
    return self._X_test

  def y_test(self):
    return self._y_test

  def labels(self):
    return self._labels
  
  def channels(self):
    return 3
  
  def dim(self):
    return 32  

class Cifar100Dataset(ImageData):

  def __init__(self, path, channels_first=True):
    super(Cifar100Dataset, self).__init__()

    # Training data
    training_data = pickle.load(open(os.path.join(path, "train"), 'rb'), encoding='latin1')

    self._y_train = np.asarray(training_data['fine_labels'], int).tolist() # train on fine labels rather than coarse

    # Test data
    testing_data = pickle.load(open(os.path.join(path, "test"), 'rb'), encoding='latin1')
    
    self._y_test = np.asarray(testing_data['fine_labels'], int).tolist()

    if channels_first:
      self._X_train = training_data['data'].reshape(-1, 3, 32, 32)
      self._X_test = testing_data['data'].reshape(-1, 3, 32, 32)
    else:
      self._X_train = training_data['data'].reshape(-1, 32, 32, 3)
      self._X_test = testing_data['data'].reshape(-1, 32, 32, 3)

    # Labels
    classes = pickle.load(open(os.path.join(path, "meta"), 'rb'), encoding='latin1')
    self._labels = classes['fine_label_names']

    # Normalize X training data
    mean = self._X_train.mean(axis=(0,2,3), keepdims=True).astype(float)
    std = self._X_train.std(axis=(0,2,3), keepdims=True).astype(float)

    self._X_train = ((self._X_train - mean) / std).tolist()
    self._X_test = ((self._X_test - mean) / std).tolist()

    print("loading complete")
  
  # Implement ImageData interface
    
  def X_train(self):
    return self._X_train
  
  def y_train(self):
    return self._y_train

  def X_test(self):
    return self._X_test

  def y_test(self):
    return self._y_test

  def labels(self):
    return self._labels
  
  def channels(self):
    return 3
  
  def dim(self):
    return 32
  
class MnistDataset(ImageData):
  def __init__(self, path, channels_first=True):

    if channels_first:
      self._X_train = MNIST(root=path, train=True, download=True).data.numpy().reshape(-1, 1, 28, 28).astype(float).tolist()
      self._X_test = MNIST(root=path, train=False, download=True).data.numpy().reshape(-1, 1, 28, 28).astype(float).tolist()
    else:
      self._X_train = MNIST(root=path, train=True, download=True).data.numpy().reshape(-1, 28, 28, 1).astype(float).tolist()
      self._X_test = MNIST(root=path, train=False, download=True).data.numpy().reshape(-1, 28, 28, 1).astype(float).tolist()
    
    self._y_train = MNIST(root=path, train=True, download=True).targets.numpy().tolist()
    self._y_test = MNIST(root=path, train=False, download=True).targets.numpy().tolist()

    self._labels = list(range(10))

  # Implement ImageData interface
    
  def X_train(self):
    return self._X_train
  
  def y_train(self):
    return self._y_train

  def X_test(self):
    return self._X_test

  def y_test(self):
    return self._y_test

  def labels(self):
    return self._labels
  
  def channels(self):
    return 1
  
  def dim(self):
    return 28