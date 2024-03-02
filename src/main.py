import argparse
import imagedata
from pytorch import cnn as pyt_cnn
from mlx import cnn as mlx_cnn

DEFAULT_INIT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_TRAINING_SPLIT = 0.9
DEFAULT_VALIDATION_SPLIT = 1.0 - DEFAULT_TRAINING_SPLIT

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int, required=True, help='training batch size')
  parser.add_argument('--bench', action='store_true', help='enable or disable time benchmark recording (default: true)')
  parser.add_argument('--cifar10', type=str, required=False, help='path to CIFAR-10 dataset directory')
  parser.add_argument('--cifar100', type=str, required=False, help='path to CIFAR-100 dataset directory')
  parser.add_argument('--device', type=str, required=False, help='specify device for workload [cpu | gpu]')
  parser.add_argument('--epochs', type=int, required=False, help='number of training epochs; defaults to 10')
  parser.add_argument('--lr', type=float, required=False, help='learning rate')
  parser.add_argument('--framework', type=str, required=True, help='framework to execute [pytorch | mlx]')
  parser.add_argument('--mnist', type=str, required=False, help='path to MNIST dataset directory')
  args = parser.parse_args()

  channels_first = args.framework=="pytorch"

  if args.cifar10:
    data = imagedata.Cifar10Dataset(args.cifar10)
  elif args.cifar100:
    data = imagedata.Cifar100Dataset(args.cifar100, channels_first=channels_first)
  elif args.mnist:
    data = imagedata.MnistDataset(args.mnist, channels_first=channels_first)
  
  if args.framework == "pytorch":
    pyt_cnn.train_pytorch_cnn(args, data)
  elif args.framework == "mlx":
    mlx_cnn.train_mlx_cnn(args, data)
  else:
    print(f"No ML framework specified")
    exit(1)

if __name__ == '__main__':
  main()