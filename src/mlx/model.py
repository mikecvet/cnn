import math
from mlx.nn import Dropout
from mlx.nn import Module
from mlx.nn import Conv2d
from mlx.nn import Linear
from mlx.nn import MaxPool2d
from mlx.nn import AvgPool2d
from mlx.nn import ReLU
from mlx.nn import LeakyReLU
from mlx.nn import LogSoftmax
import mlx.core as mx
import mlx.nn as nn

class CNN(Module):
  """
  A custom Convolutional Neural Network (CNN) designed for image classification tasks, 
  built using PyTorch's nn.Module. This network architecture can be applied to a variety of 
  image datasets such as CIFAR and MNIST (where it is easily able to achieve 99%+ accuracy), 
  though it underperforms compared to more sophisticated model architectures such as Resnet. 

  The model consists of several convolutional layers followed by max pooling to extract 
  features, and fully connected layers for classification. Dropout layers are included to 
  reduce overfitting.

  Parameters:
  - channels (int): The number of input channels (e.g., 1 for grayscale images, 3 for RGB images).
  - dim (tuple): The dimensions of the input images (height, width). It is assumed 
    that the height and width are equal.
  - classes (int): The number of output classes for the classification task.

  The network architecture is as follows:
  - First block:  Convolutional layer with ReLU activation followed by MaxPooling.
  - Second block: Another Convolutional layer with ReLU activation followed by MaxPooling.
  - Third block:  A third Convolutional layer with ReLU activation followed by MaxPooling.
  - Fourth block: A Linear layer with dropout and ReLU activation for the first fully connected layer.
  - Fifth block:  A second Linear layer with dropout for reducing the feature size further.
  - Sixth block:  A third Linear layer with dropout leading to the final classification layer.
  - Output layer: A LogSoftmax layer for producing the output probabilities for each class.

  Dropout is applied after each fully connected layer with a dropout probability of 0.3 
  to prevent overfitting. The model automatically calculates the input size for the 
  first fully connected layer based on the input image dimensions.
  """
    
  def __init__(self, channels, dim, classes):
    """
    Initializes this network, as described above. Dynamically determines some of the input feature 
    dimensionality of the later linear layers, based on the (assumed square) height or width 
    of the image data.
    """
    super(CNN, self).__init__()

    final_out_channels = 100
    fully_connected_input_size = self.calculate_fc_input_size(dim, final_out_channels)

    # First block: Conv => ReLU => MaxPool
    self.conv1 = Conv2d(in_channels=channels, out_channels=20, kernel_size=(5, 5), padding=2)
    self.relu1 = ReLU()
    self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    # Second block: Conv => ReLU => MaxPool
    self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), padding=2)
    self.relu2 = ReLU()
    self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    # Third block: Conv => ReLU => MaxPool layers
    self.conv3 = Conv2d(in_channels=50, out_channels=final_out_channels, kernel_size=(5, 5), padding=2)
    self.relu3 = ReLU()
    self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    # Fourth block: Linear => Dropout => ReLU layers
    self.linear1 = Linear(input_dims=fully_connected_input_size, output_dims=fully_connected_input_size // 2)
    self.dropout1 = Dropout(p=0.3)
    self.relu3 = ReLU()

    # Fifth block: Linear => Dropout layers
    self.linear2 = Linear(input_dims=fully_connected_input_size // 2, output_dims=fully_connected_input_size // 4)
    self.dropout2 = Dropout(p=0.3)

    # Sixth block: Linear => Dropout layers
    self.linear3 = Linear(input_dims=fully_connected_input_size // 4, output_dims=classes)
    self.dropout3 = Dropout(p=0.3)

    #self.logSoftmax = LogSoftmax(dim=1)

  def __call__(self, X):
    # First block: Conv => ReLU => MaxPool
    X = self.conv1(X)
    X = self.relu1(X)
    X = self.maxpool1(X)

    # Second block: Conv => ReLU => MaxPool
    X = self.conv2(X)
    X = self.relu2(X)
    X = self.maxpool2(X)

    # Third block: Conv => ReLU => MaxPool layers
    X = self.conv3(X)
    X = self.relu3(X)
    X = self.maxpool3(X)

    # Flatten second layer outputs to fit into the fully-connected layer `linear1`
    X = mx.flatten(X, 1)

    # Fourth block: Linear => Dropout => ReLU layers
    X = self.linear1(X)
    X = self.dropout1(X)
    X = self.relu3(X)

    # Fifth block: Linear => Dropout layers
    X = self.linear2(X)
    X = self.dropout2(X)

    # Sixth block: Linear => Dropout layers
    X = self.linear3(X)
    X = self.dropout3(X)

    # Generate class predictions
    return nn.log_softmax(X, axis=1)
  
  def calculate_fc_input_size(self, square_dim, final_out_channels, num_pools=3):
      """
      Calculate the input feature dimension for the first fully-connected layer
      based on the input image dimensionality.

      Args:
      - square_dim (int): The width and height of the square input images.
      - num_convs (int): Number of convolutional layers (assumed to preserve dimensions).
      - num_pools (int): Number of pooling layers (assumed to halve dimensions each time).

      Returns:
      - int: The number of input features for the first fully-connected layer.
      """
      # Assuming each convolutional layer is followed by a pooling layer that halves the input dimensions
      # and that convolutions are set up to preserve dimensions.
      final_size = square_dim // (2 ** num_pools)

      # Assuming the output channel size of the last convolutional layer is fixed.
      # For example, if the last conv layer outputs 100 channels:
      output_channels = final_out_channels

      # Calculate the total number of input features for the first fully-connected layer.
      fc_input_features = int(final_size * final_size * output_channels)

      return fc_input_features

