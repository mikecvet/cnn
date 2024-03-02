import math
import torch
from torch.utils.data import Dataset, random_split

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
  def __init__(self, data, device, training_split):
    training_dataset = TrainingData(data, device)
    self.test_dataset = TestingData(data, device)
    training_samples = int(len(training_dataset) * training_split)
    validation_samples = int(math.ceil(len(training_dataset) * (1.0 - training_split)))
    self.training_data, self.validation_data = random_split(training_dataset, [training_samples, validation_samples])

class TrainingData(Dataset):
  def __init__(self, data, device):
    super(TrainingData, self).__init__()
    self.X = torch.tensor(data.X_train(), dtype=torch.float).to(device)
    self.y = torch.tensor(data.y_train(), dtype=torch.long).to(device)
    
  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
  
class TestingData(Dataset):
  def __init__(self, data, device):
    super(TestingData, self).__init__()
    self.X = torch.tensor(data.X_test(), dtype=torch.float).to(device)
    self.y = torch.tensor(data.y_test(), dtype=torch.long).to(device)
    
  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]