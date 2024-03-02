from . import dataset, model
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import sys
import time
from tqdm import tqdm

DEFAULT_INIT_LR = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_TRAINING_SPLIT = 0.9
DEFAULT_VALIDATION_SPLIT = 1.0 - DEFAULT_TRAINING_SPLIT

def train_mlx_cnn(args, data, batch_size):

  if args.bench:
    log = { "inference": [], "backprop": [], "epoch": [], "precision": [], "loss": [] }
  else:
    log = {}

  ds = dataset.SplitDataset(data, training_split=DEFAULT_TRAINING_SPLIT)
  training_data_loader = dataset.DataLoader(ds.training_data, batch_size=batch_size)
  validation_data_loader = dataset.DataLoader(ds.validation_data, batch_size=batch_size)
  test_data_loader = dataset.DataLoader(ds.test_dataset, batch_size=1)

  cnn = model.CNN(data.channels(), data.dim(), len(data.labels()))
  train(training_data_loader, validation_data_loader, cnn, args.epochs or 10, batch_size, args.lr or DEFAULT_INIT_LR, log)
  test(test_data_loader, cnn, log)

'''
def time_predict(model, X, log):
  t0 = time.perf_counter_ns()
  p = model(X)
  ns = (time.perf_counter_ns() - t0)

  if "inference" in log:
    log["inference"].append(ns)

  return p
  '''

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
    if log["inference"]:
      min, avg, max = find_range_us(log["inference"])
      f.write(f"batch_inference_range_us|{batch_size}:{min},{avg},{max}\n")
      f.write(f"indiv_inference_range_us:{min//batch_size},{avg//batch_size},{max//batch_size}\n")

    if log["backprop"]:
      min, avg, max = find_range_us(log["backprop"])
      f.write(f"batch_backprop_range_us|{batch_size}:{min},{avg},{max}\n")
      f.write(f"indiv_backprop_range_us:{min//batch_size},{avg//batch_size},{max//batch_size}\n")

    if log["epoch"]:
      min, avg, max = find_range_us(log["epoch"])
      f.write(f"epoch_range_s|{batch_size}:{min/1000000:.3f},{avg/1000000:.3f},{max/1000000:.3f}\n")

def loss_fn(p, y):
  return nn.losses.nll_loss(p, y)

def train(training_data_loader, validation_data_loader, model, epochs, batch_size, learning_rate, log):
  optimizer = optimizers.Adam(learning_rate)
  state = [model.state, optimizer.state]

  def train_step(model, inp, tgt):
    output = model(inp)
    loss = mx.mean(nn.losses.nll_loss(output, tgt))
    acc = mx.mean(mx.argmax(output, axis=1) == tgt)
    return loss, acc
  
  @partial(mx.compile, inputs=state, outputs=state)
  def step(inp, tgt):
      train_step_fn = nn.value_and_grad(model, train_step)
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

  mx.eval(model.parameters())

  for i in range(epochs):
    train_start = time.perf_counter_ns()
    total_loss = 0.0
    correct = 0
  
    # Set training mode
    model.train() 
    
    # Number of training batches to process
    n = len(training_data_loader.dataset) // batch_size + 1

    # tqdm renders a progress bar while these batches are processed
    for _, (X, y) in tqdm(enumerate(training_data_loader), total=n):
      #p = time_predict(model, X, log)

      loss, acc = time_predict(model, X, y, log)

      mx.eval(state)
      
      #mx.eval(model.parameters(), optimizer.state)
      #loss = time_backprop(loss_fn, p, y, optimizer, log)

      #print(f"total loss: {total_loss}")
      #total_loss += loss.item()
      #correct += (p.argmax(axis=1) == y).type(mx.float).sum().item()
      #correct += acc.item()

    train_end = time.perf_counter_ns()
    avg_loss = total_loss / len(training_data_loader)
    precision = (correct / len(training_data_loader.dataset)) * 100
    print(f"training epoch {i + 1} avg loss: {avg_loss} accuracy: {precision:0.3f}%")

    total_loss = 0.0
    correct = 0

    val_start = time.perf_counter_ns()

    model.eval()
    # Disable gradient calculations during inference workloads
    #with mx.stop_gradient():

      # Set evaluation mode
      #model.eval()

      # Number of validation batches to process
    n = len(validation_data_loader.dataset) // batch_size + 1

    for _, (X, y) in tqdm(enumerate(validation_data_loader), total=n):
      #with mx.stop_gradient(X):
      #p = time_predict(model, X, log)
      #total_loss += loss_fn(p, y)
      loss, acc = train_step(model, X, y)
      #total_loss += loss.item()
      #correct += acc

      # Calculate the number of corerect predictions
      #correct += (p.argmax(1) == y).type(mx.float).sum().item() 
      #correct += (p.argmax(1) == y).sum().item()
    
    val_end = time.perf_counter_ns()
    avg_loss = total_loss / len(validation_data_loader)
    precision = (correct / len(validation_data_loader.dataset)) * 100
    #print(f"validation epoch {i + 1} avg loss: {avg_loss} precision: {precision.item():0.3f}%")

    log["epoch"].append((train_end - train_start) + (val_end - val_start))

  save_benchmarks(log, batch_size)

def test(test_data_loader, model, log):
  accuracy = 0
  predictions = []

  print ("testing ...")

  def train_step(model, inp, tgt):
    output = model(inp)
    loss = mx.mean(nn.losses.nll_loss(output, tgt))
    acc = mx.mean(mx.argmax(output, axis=1) == tgt)
    return loss, acc

  # Disable gradient calculations during inference workloads
  #with mx.stop_gradient():

    # Set evaluation mode
  model.eval()
  total_loss = 0.0
  accuracy = 0

  for _, (X, y) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
    #p = time_predict(model, X, log)
    loss, acc = train_step(model, X, y)
    total_loss += loss
    accuracy += acc

    # Extract class prediction
    #klass = p.argmax(dim=1)
    #predictions.append(klass.item())
    #accuracy += (klass == y).type(mx.float).sum().item() 
    
  print(f"Final test dataset accuracy: {(accuracy.item() / len(test_data_loader)) * 100:0.3f}%")