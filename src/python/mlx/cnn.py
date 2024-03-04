from . import dataset, model
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import sys
import time
from tqdm import tqdm

import pstats
import cProfile

DEFAULT_INIT_LR = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_TRAINING_SPLIT = 0.9
DEFAULT_VALIDATION_SPLIT = 1.0 - DEFAULT_TRAINING_SPLIT

def train_mlx_cnn(args, data):

  if args.bench:
    log = { "inference": [], "backprop": [], "epoch": [], "test": [] }
  else:
    log = {}

  batch_size = args.batch or DEFAULT_BATCH_SIZE
  ds = dataset.SplitDataset(data, training_split=DEFAULT_TRAINING_SPLIT)
  training_data_loader = dataset.DataLoader(ds.training_data, batch_size=batch_size)
  validation_data_loader = dataset.DataLoader(ds.validation_data, batch_size=batch_size)
  test_data_loader = dataset.DataLoader(ds.test_dataset, batch_size=1)

  with cProfile.Profile() as profile:
    cnn = model.CNN(data.channels(), data.dim(), len(data.labels()))

    train(training_data_loader, validation_data_loader, cnn, args.epochs or DEFAULT_EPOCHS, batch_size, args.lr or DEFAULT_INIT_LR, log)
    test(test_data_loader, cnn, log)
  profile_result = pstats.Stats(profile)
  profile_result.sort_stats(pstats.SortKey.TIME)
  profile_result.print_stats()


def time_backprop(loss_fn, p, y, optimizer, log):
  t0 = time.perf_counter_ns()
  loss = loss_fn(p, y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ns = (time.perf_counter_ns() - t0)

  if "backprop" in log:
    log["backprop"].append(ns)

  return loss

def find_range_us(L):
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

def save_benchmarks(log, batch_size):
  with open("cnn.bench.out", "a") as f:
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
      test_time = log["test"]
      f.write(f"test_example_item_us:{test_time/1000:.3f}n")  

def loss_fn(p, y):
  return nn.losses.nll_loss(p, y)

def loss_and_accuracy(model, inp, tgt):
  output = model(inp)
  loss = mx.mean(nn.losses.nll_loss(output, tgt))
  acc = mx.mean(mx.argmax(output, axis=1) == tgt)
  return loss, acc

def train(training_data_loader, validation_data_loader, model, epochs, batch_size, learning_rate, log):
  optimizer = optimizers.SGD(learning_rate)
  state = [model.state, optimizer.state]

  mx.eval(model.parameters())

  @partial(mx.compile, inputs=state, outputs=state)
  def step(inp, tgt):
      train_step_fn = nn.value_and_grad(model, loss_and_accuracy)
      (loss, acc), grads = train_step_fn(model, inp, tgt)
      optimizer.update(model, grads)
      return loss, acc
  
  def time_predict(model, X, y, log):
    t0 = time.perf_counter_ns()
    loss, acc = step(X, y)
    ns = (time.perf_counter_ns() - t0)

    if "inference" in log:
      log["inference"].append(ns)

    return loss, acc

  total_loss = 0.0

  for i in range(epochs):
    train_start = time.perf_counter_ns()
    total_loss = 0.0
    accuracy = 0
  
    # Set training mode
    model.train() 
    
    # Number of training batches to process
    n = len(training_data_loader.dataset) // batch_size + 1

    # Train over batches
    # tqdm renders a progress bar while these batches are processed
    for _, (X, y) in tqdm(enumerate(training_data_loader), total=n):
      loss, acc = time_predict(model, X, y, log)
      total_loss += loss
      accuracy += acc

      #mx.eval(state, total_loss, accuracy)

    train_end = time.perf_counter_ns()
    
    avg_loss = total_loss.item() / len(training_data_loader)
    accuracy_pct = (accuracy.item() / len(training_data_loader.dataset)) * 100
    print(f"training epoch {i + 1} avg loss: {avg_loss} accuracy: {accuracy_pct:0.3f}%")

    # Validation with unseen test data
    val_start = time.perf_counter_ns()
    accuracy, total_loss = eval(validation_data_loader, model, len(validation_data_loader.dataset) // batch_size + 1)
    val_end = time.perf_counter_ns()

    avg_loss = total_loss.item() / len(validation_data_loader)
    accuracy_pct = (accuracy.item() / len(validation_data_loader.dataset)) * 100
    print(f"validation epoch {i + 1} avg loss: {avg_loss} accuracy: {accuracy_pct:0.3f}%")

    if "epoch" in log:
      log["epoch"].append((train_end - train_start) + (val_end - val_start))

  save_benchmarks(log, batch_size)

def test(test_data_loader, model, log):
  print ("testing ...")

  test_start = time.perf_counter_ns()
  accuracy, _ = eval(test_data_loader, model, len(test_data_loader))
  test_end = time.perf_counter_ns() 

  if "test" in log:
    log["test"].append(test_end - test_start)

  print(f"Final test dataset accuracy: {(accuracy.item() / len(test_data_loader)) * 100:0.3f}%")

def eval(data_loader, model, n):
  """
  Evaluates the performance of the CNN model on a given dataset.

  This function calculates the loss and accuracy of the model using the provided data loader. It sets the model
  to evaluation mode, disables gradient calculations, and iterates through the dataset to compute the total loss
  and the number of correct predictions.

  Parameters:
  - data_loader (DataLoader): A PyTorch DataLoader instance containing the dataset for evaluation.
  - model (Module): The CNN model to be evaluated.
  - log (dict): A dictionary for logging evaluation performance metrics.

  Returns:
  - correct (int): The total number of correct predictions made by the model on the dataset.
  - total_loss (float): The total loss computed across all batches in the dataset.

  Note:
  The function uses a helper function `time_predict` to make predictions with the model and log the time taken
  for these predictions. It assumes the loss function used during model training is nn.NLLLoss.
  """
  accuracy = 0
  total_loss = 0.0
  accuracy = 0

  # Set evaluation mode
  model.eval()

  for _, (X, y) in tqdm(enumerate(data_loader), total=n):
    loss, acc = loss_and_accuracy(model, X, y)
    total_loss += loss
    accuracy += acc

  return accuracy, total_loss