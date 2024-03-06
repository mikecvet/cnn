# CNN

This repository contians parallel implementations of a bespoke convolutional neural network leveraging both PyTorch and MLX. 

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
