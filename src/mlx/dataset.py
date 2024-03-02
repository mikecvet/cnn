import mlx.core as mx
import random

class SplitDataset:
  def __init__(self, data, training_split):
    X_train, y_train, X_validation, y_validation = split_training_data(data.X_train(), data.y_train(), training_split)
    self.training_data = Dataset(X_train, y_train)
    self.validation_data = Dataset(X_validation, y_validation)
    self.test_dataset = Dataset(data.X_test(), data.y_test())

class Dataset:
  def __init__(self, X, y):
    #print(f"type: {type(X)} data: {X[0:100]}")
    self.X = mx.array(X)
    self.y = mx.array(y)
    #self.X = X
    #self.y = y
    
  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
  
class DataLoader:
  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size
  
  def __iter__(self):
    for batch_id in range(0, len(self.dataset), self.batch_size):
      yield self.dataset[batch_id:batch_id+self.batch_size]
  
  def __len__(self):
    return len(self.dataset) // self.batch_size

def split_training_data(X, y, training_split):
  paired = list(zip(X, y))
  random.shuffle(paired)

  X, y = zip(*paired)
  #print(f"X type: {type(X)}:{type(X[0])}{type(X[0][0])}{type(X[0][0][0])}{type(X[0][0][0][0])}")
  X = mx.array(X) # should the inner lists be confered as well?
  y = mx.array(y)
  print(f"dataset: X {X.shape} y {y.shape}")
  #X = convert_to_mlx(X)
  #y = convert_to_mlx(y)
  #print(f"len(X) {len(X)} type {type(X)} type X[0] {type(X[0])} type X[0][0] {type(X[0][0])}  type X[0][0][0] {type(X[0][0][0])} type X[0][0][0][0] {type(X[0][0][0][0])}")

  split_index = int(training_split * len(X))

  X_train = X[:split_index]
  y_train = y[:split_index]
  X_validation = X[split_index:]
  y_validation = y[split_index:]

  return X_train, y_train, X_validation, y_validation

import numpy as np
# Assuming mlx.array exists and works similarly to how numpy arrays are converted
import mlx

def convert_to_mlx(nested_np_array, convert_primitive_types=False, depth=0):
    """
    Recursively converts a nested structure of numpy arrays into equivalent MLX arrays.
    
    Args:
    nested_np_array: The nested numpy array structure to convert.
    
    Returns:
    The equivalent nested MLX array structure.
    """
    # Base case: if the current element is a numpy array, convert it directly
    if isinstance(nested_np_array, np.ndarray):
        print(f"returning mx.array {depth}")
        return mx.array(nested_np_array)
    # Recursive case: iterate through items if it's an iterable (list or similar)
    elif isinstance(nested_np_array, (list, tuple)) and isinstance(nested_np_array[0], list):
        print(f"iterating deeper {depth}")
        return type(nested_np_array)(convert_to_mlx(item, depth=depth+1) for item in nested_np_array)
    else:
      print(f"base item {depth}")
      return mx.array(nested_np_array)