# Parallel Implementations of a Convolutional Neural Network with PyTorch and Apple's MLX

This repository contains parallel implementations of a bespoke convolutional neural network leveraging both [PyTorch](https://pytorch.org/docs/stable/index.html) and [MLX](https://ml-explore.github.io/mlx/build/html/index.html).

This code supports running training and classification against the MNIST, CIFAR-10, and CIFAR-100 datasets, via the [ImageData](https://github.com/mikecvet/cnn/blob/main/src/python/imagedata.py) class. This is then used in both implementations [1](https://github.com/mikecvet/cnn/tree/main/src/python/pytorch) [2](https://github.com/mikecvet/cnn/tree/main/src/python/mlx) to organize training, validation and testing datasets. This model should be able to hit 99.1%+ accuracy on the MNIST dataset, and 30-40%+ accuracy on the CIFAR datasets.

<img width="1093" alt="acc_and_loss" src="https://github.com/mikecvet/cnn/assets/275631/35f965e8-b441-433a-a95f-30ff8bf4e8d0">

Here is the model architecture, taken from [src/python/pytorch/model.py](https://github.com/mikecvet/cnn/blob/main/src/python/pytorch/model.py): 

```
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
    self.linear1 = Linear(in_features=fully_connected_input_size, out_features=fully_connected_input_size // 2)
    self.dropout1 = Dropout(p=0.3)
    self.relu3 = ReLU()

    # Fifth block: Linear => Dropout layers
    self.linear2 = Linear(in_features=fully_connected_input_size // 2, out_features=fully_connected_input_size // 4)
    self.dropout2 = Dropout(p=0.3)

    # Sixth block: Linear => Dropout layers
    self.linear3 = Linear(in_features=fully_connected_input_size // 4, out_features=classes)
    self.dropout3 = Dropout(p=0.3)

    self.logSoftmax = LogSoftmax(dim=1)
```

An example run here, which trains a new pytorch-based model on the MNIST dataset, and then tests its accuracy after 10 epochs:
```
   ~>> python3 main.py --mnist ./mnist --epochs 10 --device cpu --batch 512 --lr 0.00001 --framework mlx --bench
  100%|███████████████████████████████████████████████████████████████████| 106/106 [00:21<00:00,  4.93it/s]
  training epoch 1 avg loss: 1.3816425141834077 accuracy: 54.744%
  100%|██████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 10536.25it/s]
  validation epoch 1 accuracy: 92.417%
  100%|███████████████████████████████████████████████████████████████████| 106/106 [00:27<00:00,  3.83it/s]
  [...]
  training epoch 9 avg loss: 0.4642796471005394 accuracy: 78.893%
  100%|██████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 11823.27it/s]
  validation epoch 9 accuracy: 97.283%
  100%|███████████████████████████████████████████████████████████████████| 106/106 [00:23<00:00,  4.42it/s]
  training epoch 10 avg loss: 0.4539340064639137 accuracy: 79.154%
  100%|███████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 9388.48it/s]
  validation epoch 10 accuracy: 97.317%
  testing ...
  100%|████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 13943.48it/s]
  Final test dataset accuracy: 97.800% loss: 0.07119915771484375
```
