from . import dataset, model
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

DEFAULT_INIT_LR = 1e-4
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 10
DEFAULT_TRAINING_SPLIT = 0.9
DEFAULT_VALIDATION_SPLIT = 1.0 - DEFAULT_TRAINING_SPLIT

def train_pytorch_cnn(args, data):
  """
  Trains a pytorch-based CNN model based on given arguments and data. Creates training, validation and test datasets, and then
  runs training and evaluation based on that data.
  """
  if args.bench:
    # Initialize the log dict with entries for tracking
    log = { "inference": [], "backprop": [], "epoch": [], "test": [], "loss": [], "accuracy": []}
  else:
    log = {}

  batch_size = args.batch or DEFAULT_BATCH_SIZE
  device = detect_device(args)

  ds = dataset.SplitDataset(data, device, training_split=DEFAULT_TRAINING_SPLIT)
  training_data_loader = DataLoader(ds.training_data, shuffle=True, batch_size=batch_size)
  validation_data_loader = DataLoader(ds.validation_data, shuffle=True, batch_size=batch_size)
  test_data_loader = DataLoader(ds.test_dataset, shuffle=False, batch_size=1)

  cnn = model.CNN(data.channels(), data.dim(), len(data.labels())).to(device)
  train(training_data_loader, validation_data_loader, cnn, args.epochs or DEFAULT_EPOCHS, batch_size or DEFAULT_BATCH_SIZE, args.lr or DEFAULT_INIT_LR, log)
  test(test_data_loader, cnn, log)

  return log

def detect_device(args):
  """
  Depending on the provided arguments and the available device backends, returns either a CPU,
  or GPU device, where the GPU device may be either a CUDA-based GPU or Apple Silicon "MPS" based GPU.
  Default device is CPU.
  """
  if args.device:
    if args.device == "gpu" and torch.backends.mps.is_available():
      return torch.device("mps")
    elif args.device == "gpu" and torch.backends.cuda.is_available:
      return torch.device("cuda")
    else:
      return torch.device("cpu")  # Fallback to CPU if MPS is not available
  else:
    return torch.device("cpu")  # CPU by default
  
def train(training_data_loader, validation_data_loader, model, epochs, batch_size, learning_rate, log):
  """
  Trains and validates the given CNN model using the specified training and validation data loaders.

  The function iteratively trains the model for a given number of epochs, using the Adam optimizer and 
  negative log likelihood loss function. It tracks and logs the training and validation loss, as well as 
  the accuracy for each epoch. The performance metrics are printed out at the end of each epoch for both 
  training and validation phases.

  Parameters:
  - training_data_loader (DataLoader): A PyTorch DataLoader instance containing the training dataset.
  - validation_data_loader (DataLoader): A PyTorch DataLoader instance containing the validation dataset.
  - model (Module): The CNN model instance to be trained and validated.
  - epochs (int): The number of complete passes through the dataset.
  - batch_size (int): The number of samples per gradient update. This parameter is used to calculate 
    the number of training batches.
  - learning_rate (float): The learning rate for the Adam optimizer.
  - log (dict): A dictionary for logging training and validation performance metrics. If tracking benchmark
  data is enabled, it is expected to have a key "epoch" with an empty list as its value, which will be 
  populated with epoch durations.

  The function performs the following operations for each epoch:
  1. Sets the model to training mode.
  2. Iterates over the training dataset, computing the loss for each batch and updating the model parameters.
  3. Calculates and logs the average training loss and accuracy.
  4. Sets the model to evaluation mode and performs validation on the validation dataset.
  5. Calculates and logs the average validation loss and accuracy.

  Additionally, the function logs the time taken for each training and validation epoch, adding these 
  durations to the provided log dictionary under the "epoch" key.

  Note: The function assumes the presence of helper functions `time_predict` and `time_backprop` for 
  making predictions and performing backpropagation, respectively, while logging the time taken for 
  these operations. The `eval` function performs the evaluation on the validation dataset on unseen 
  training data.
  """

  optimizer = torch.optim.Adam(model.parameters(), learning_rate)
  loss_fn = nn.NLLLoss()

  total_loss = 0.0

  for i in range(epochs):
    total_loss = 0.0
    accuracy = 0
  
    # Set training mode
    model.train()

    # Number of training batches to process
    n = len(training_data_loader.dataset) // batch_size + 1

    train_start = time.perf_counter_ns()

    # tqdm renders a progress bar while these batches are processed
    for _, (X, y) in tqdm(enumerate(training_data_loader), total=n):
      p = time_predict(model, X, log)
      loss = time_backprop(loss_fn, p, y, optimizer, log)
      total_loss += loss
      accuracy += (p.argmax(axis=1) == y).type(torch.float).sum().item()

    train_end = time.perf_counter_ns()

    avg_loss = total_loss / len(training_data_loader)
    accuracy_pct = (accuracy / len(training_data_loader.dataset)) * 100
    print(f"training epoch {i + 1} avg loss: {avg_loss} accuracy: {accuracy_pct:0.3f}%")

    total_loss = 0.0
    accuracy = 0

    # Executes a validation step of unseen training data
    accuracy, _ = eval(validation_data_loader, model, log)
    accuracy_pct = (accuracy / len(validation_data_loader.dataset)) * 100

    if "epoch" in log:
      log["epoch"].append(train_end - train_start)
      log["loss"].append(avg_loss.item())
      log["accuracy"].append(accuracy_pct)

    print(f"validation epoch {i + 1} accuracy: {accuracy_pct:0.3f}%")

def test(data_loader, model, log):
  """
  Conducts a test to evaluate the CNN model's performance on a test dataset.

  This function leverages the `eval` function to calculate the accuracy of the model on the given dataset,
  logging the time taken to complete the test. It updates the log dictionary with the duration of the test
  execution. Finally, it prints out the model's accuracy on the test dataset.

  Parameters:
  - data_loader (DataLoader): A DataLoader instance containing the test dataset.
  - model (Module): The CNN model to be tested.
  - log (dict): A dictionary for logging performance metrics. If benchmarking is enabled, the dictionary should 
    have a key "test" where the test execution time will be appended.

  The function calculates the test accuracy by dividing the total number of correct predictions by the
  total number of items in the dataset, multiplying by 100 to get a percentage. This accuracy is then printed
  to the console.
  """  
  print ("testing ...")

  test_start = time.perf_counter_ns()
  accuracy, _ = eval(data_loader, model, log)
  test_end = time.perf_counter_ns()

  if "test" in log:
    log["test"].append(test_end - test_start)

  print(f"Final test dataset accuracy: {(accuracy / len(data_loader)) * 100:0.3f}%")

def eval(data_loader, model, log):
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
  loss_fn = nn.NLLLoss()

  # Set evaluation mode
  model.eval()
  
  # Disable gradient calculations during inference workloads
  with torch.no_grad():
    for _, (X, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
      p = time_predict(model, X, log)
      total_loss += loss_fn(p, y)

       # Calculate the number of correct predictions
      accuracy += (p.argmax(1) == y).type(torch.float).sum().item() 

    return accuracy, total_loss

def time_predict(model, X, log):
  """
  Runs inference and tracks execution time for the given mdoel.
  """
  t0 = time.perf_counter_ns()
  p = model(X)
  ns = (time.perf_counter_ns() - t0)

  if "inference" in log:
    log["inference"].append(ns)

  return p

def time_backprop(loss_fn, p, y, optimizer, log):
  """
  Runs calculates loss, runs backpropagation and tracks execution time for the given mdoel.
  """
  t0 = time.perf_counter_ns()
  loss = loss_fn(p, y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ns = (time.perf_counter_ns() - t0)

  if "backprop" in log:
    log["backprop"].append(ns)

  return loss
