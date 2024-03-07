from . import dataset, model
from functools import partial
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import time
from tqdm import tqdm

DEFAULT_INIT_LR = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_TRAINING_SPLIT = 0.9
DEFAULT_VALIDATION_SPLIT = 1.0 - DEFAULT_TRAINING_SPLIT

def train_mlx_cnn(args, data, log):
  """
  Trains an MLX-based CNN model based on given arguments and data. Creates training, validation and test datasets, and then
  runs training and evaluation based on that data.
  """
  batch_size = args.batch or DEFAULT_BATCH_SIZE
  
  ds = dataset.SplitDataset(data, training_split=DEFAULT_TRAINING_SPLIT)
  training_data_loader = dataset.DataLoader(ds.training_data, batch_size=batch_size)
  validation_data_loader = dataset.DataLoader(ds.validation_data, batch_size=batch_size)
  test_data_loader = dataset.DataLoader(ds.test_dataset, batch_size=batch_size)

  cnn = model.CNN(data.channels(), data.dim(), len(data.labels()))

  train(training_data_loader, validation_data_loader, cnn, args.epochs or DEFAULT_EPOCHS, batch_size, args.lr or DEFAULT_INIT_LR, log)
  test(test_data_loader, cnn, batch_size, log)
  
  return log

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
  """
  optimizer = optimizers.Adam(learning_rate)
  state = [model.state, optimizer.state, mx.random.state]

  mx.eval(model.parameters())

  @partial(mx.compile, inputs=state, outputs=state)
  def step(X, y):
    """
    Performs a single training step, including forward pass, loss computation, 
    and parameter update for a given model.

    This function is decorated with `mx.compile`, which optimizes the computation
    for the specified input and output states, described further here: 
      https://ml-explore.github.io/mlx/build/html/usage/compile.html. 
    
    It computes the loss and accuracy by applying the model to the input data, calculates the 
    gradients of the model parameters with respect to the loss, and updates the model parameters 
    using the specified optimizer.

    Parameters:
    - X: Input mx.array representing the batch of data to be processed by the model.
    - y: Target mx.array containing the ground truth labels corresponding to the input data.

    Returns:
    - loss: The computed loss value as a result of comparing the model's predictions with the ground truth labels.
    - acc: The accuracy of the model's predictions, calculated as the percentage of correct predictions in the batch.
    """
    train_step_fn = nn.value_and_grad(model, loss_and_accuracy)
    (loss, acc), grads = train_step_fn(model, X, y)
    optimizer.update(model, grads)
    return loss, acc
  # End step()
  
  total_loss = 0.0

  for i in range(epochs):
    train_start = time.perf_counter_ns()
    total_loss = 0.0
    accuracy = 0
  
    # Set training mode
    model.train() 
    
    # Number of training batches to process
    n = math.ceil(len(training_data_loader.dataset) / batch_size)

    # Train over batches
    # tqdm renders a progress bar while these batches are processed
    for _, (X, y) in tqdm(enumerate(training_data_loader), total=n):
      loss, acc = step(X, y)
      total_loss += loss
      accuracy += acc

      mx.eval(state)

    train_end = time.perf_counter_ns()

    avg_loss = total_loss.item() / len(training_data_loader)
    accuracy_pct = (accuracy.item() / len(training_data_loader.dataset)) * 100

    print(f"training epoch {i + 1} avg loss: {avg_loss} accuracy: {accuracy_pct:0.3f}%")

    # Validation with unseen test data
    
    accuracy, _ = eval(validation_data_loader, model, len(validation_data_loader))
    accuracy_pct = (accuracy.item() / len(validation_data_loader.dataset)) * 100

    if "epoch" in log:
      log["epoch"].append(train_end - train_start)
      log["loss"].append(avg_loss)
      log["accuracy"].append(accuracy_pct)

    print(f"validation epoch {i + 1} accuracy: {accuracy_pct:0.3f}%")

def test(test_data_loader, model, batch_size, log):
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
  print (f"testing ...")

  test_start = time.perf_counter_ns()
  accuracy, _ = eval(test_data_loader, model, math.ceil(len(test_data_loader) / batch_size))
  mx.eval(accuracy) # Force calculation within measurement block
  test_end = time.perf_counter_ns()   

  if "test" in log:
    log["test"].append(test_end - test_start)

  print(f"Final test dataset accuracy: {(accuracy.item() / len(test_data_loader.dataset)) * 100:0.3f}%")

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

def loss_and_accuracy(model, X, y):
  """
  Computes the loss and accuracy of a model's predictions on a given set of inputs and labels.

  This function feeds forward the input tensor `X` through the `model`, calculates the negative log likelihood loss
  using the predictions `p` and the ground truth labels `y`, and evaluates the accuracy of these predictions. The
  loss is calculated as the mean negative log likelihood loss across all predictions. Accuracy is computed as the
  sum of correct predictions (where the predicted label matches the true label).

  Parameters:
  - model (Module): The model to evaluate, which should implement a forward pass method for inference.
  - X (array): The input mx.array containing the data to be fed into the model for prediction. The shape of `X`
    depends on the model's expected input dimensions.
  - y (array): The true labels corresponding to the input data `X`. The labels are expected to be a 1D mx.array
    with the same number of elements as there are samples in `X`.

  Returns:
  - loss (float): The mean negative log likelihood loss computed across all predictions in the batch.
  - acc (int): The total number of correct predictions made by the model on the input batch.
  """  
  p = model(X)
  loss = mx.mean(nn.losses.nll_loss(p, y))
  acc = mx.sum(mx.argmax(p, axis=1) == y)
  return loss, acc
