import mlx.core as mx
import random

class SplitDataset:
  """
  Splits a dataset into training, validation, and testing subsets according to the specified training split ratio.

  This class is designed to facilitate the organization of data for machine learning models by
  splitting the provided dataset into training, validation, and testing sets. It leverages custom Dataset classes for
  training and testing data to ensure compatibility with PyTorch's DataLoader for efficient batch processing during model
  training and evaluation.

  Parameters:
  - data (ImageData): An ImageData object containing the dataset
  - device (torch.device): The device (CPU or GPU) where the tensors will be allocated.
  - training_split (float): The proportion of the dataset to be used for training. The rest will be used for validation.

  Attributes:
  - training_data (Subset): A torch.utils.data.Subset instance representing the training data.
  - validation_data (Subset): A torch.utils.data.Subset instance representing the validation data.
  - test_dataset (TestingData): An instance of the TestingData class containing the testing data.
  """
  def __init__(self, data, training_split):
    X_train, y_train, X_validation, y_validation = split_training_data(data.X_train(), data.y_train(), training_split)
    self.training_data = Dataset(X_train, y_train)
    self.validation_data = Dataset(X_validation, y_validation)
    self.test_dataset = Dataset(data.X_test(), data.y_test())

class Dataset:
  def __init__(self, X, y):
    self.X = mx.array(X)
    self.y = mx.array(y)
    
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
  X = mx.array(X)
  y = mx.array(y)

  split_index = int(training_split * len(X))

  X_train = X[:split_index]
  y_train = y[:split_index]
  X_validation = X[split_index:]
  y_validation = y[split_index:]

  return X_train, y_train, X_validation, y_validation
