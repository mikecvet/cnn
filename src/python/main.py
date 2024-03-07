import argparse
import imagedata
from mlx import cnn as mlx_cnn
from pytorch import cnn as pyt_cnn
import sys

DEFAULT_BATCH_SIZE = 64

def find_range_us(L):
  """
  Given a list of what is assumed to be nanoseconds, finds the minimum and maximum elements, as well as
  calculates the average value. Returns results divided by a thousand, which returns results in 
  microseconds rather than nanoseconds.
  """
  min = sys.maxsize
  max = 0
  sum = 0

  for e in L:
    if e > max:
      max = e
    
    if e < min:
      min = e
    
    sum += e
  
  return min // 1000, sum // len(L) // 1000, max // 1000

def save_benchmarks(log, prefix, batch_size):
  """
  Write benchmark data to disk
  """
  with open(prefix + ".cnn.bench.out", "a") as f:
    if log.get("inference"):
      L = log["inference"]
      if len(L) > 1:
        L = L[1:] # Discard first element due to warmup costs

      min, avg, max = find_range_us(L) 
      f.write(f"batch_inference_range_us|{batch_size}:{min},{avg},{max}\n")
      f.write(f"indiv_inference_range_us:{min//batch_size},{avg//batch_size},{max//batch_size}\n")

    if log.get("backprop"):
      L = log["backprop"]
      if len(L) > 1:
        L = L[1:] # Discard first element due to warmup costs

      min, avg, max = find_range_us(L)
      f.write(f"batch_backprop_range_us|{batch_size}:{min},{avg},{max}\n")
      f.write(f"indiv_backprop_range_us:{min//batch_size},{avg//batch_size},{max//batch_size}\n")

    if log.get("epoch"):
      L = log["epoch"]
      if len(L) > 1:
        L = L[1:] # Discard first element due to warmup costs

      min, avg, max = find_range_us(L) # Discard first element due to warmup costs
      f.write(f"epoch_range_s|{batch_size}:{min/1000000:.3f},{avg/1000000:.3f},{max/1000000:.3f}\n")

    if log.get("test"):
      test_time = log["test"][0]
      f.write(f"test_run_ms:{test_time//1000000:.3f}\n")

    if log.get("loss") and log.get("accuracy"):
      loss = log["loss"]
      acc = log["accuracy"]
      f.write(f"loss: {loss}\n")
      f.write(f"accuracy: {acc}\n")

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
    data = imagedata.Cifar10Dataset(args.cifar10, channels_first=channels_first)
  elif args.cifar100:
    data = imagedata.Cifar100Dataset(args.cifar100, channels_first=channels_first)
  elif args.mnist:
    data = imagedata.MnistDataset(args.mnist, channels_first=channels_first)
  else:
    print("No dataset specified")
    exit(1)
  
  if args.bench:
    # Initialize the log dict with entries for tracking
    log = { "inference": [], "backprop": [], "epoch": [], "test": [], "loss": [], "accuracy": []}
  else:
    log = {}

  if args.framework == "pytorch":
    log = pyt_cnn.train_pytorch_cnn(args, data, log)
  elif args.framework == "mlx":
    log = mlx_cnn.train_mlx_cnn(args, data, log)
  else:
    print(f"No ML framework specified")
    exit(1)

  if args.bench:
    save_benchmarks(log, args.framework + "." + args.device, args.batch or DEFAULT_BATCH_SIZE)  

if __name__ == '__main__':
  main()