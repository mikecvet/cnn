import matplotlib.pyplot as plt
import argparse
import re
import numpy as np

def extract_data(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    # Adjusted regex to match any batch size
    epoch_times_matches = re.findall(r'epoch_range_s\|\d+:(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)', content)
    epoch_times = [list(map(float, match)) for match in epoch_times_matches][0] if epoch_times_matches else [0, 0, 0]

    test_time_match = re.search(r'test_run_ms:(\d+\.\d+)', content)
    test_time = float(test_time_match.group(1)) if test_time_match else 0
    
    loss_match = re.search(r'loss: \[([^\]]+)\]', content)
    accuracy_match = re.search(r'accuracy: \[([^\]]+)\]', content)
    loss = [float(x) for x in loss_match.group(1).split(', ')] if loss_match else []
    accuracy = [float(x) for x in accuracy_match.group(1).split(', ')] if accuracy_match else []

    return epoch_times, test_time, loss, accuracy

def plot_graphs(data):
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    
    # Graph 1: Epoch training time intervals
    for i, (label, values) in enumerate(data.items()):
        epoch_times, _, _, _ = values
        axs[0].errorbar(i, np.mean(epoch_times), yerr=[[np.mean(epoch_times) - min(epoch_times)], [max(epoch_times) - np.mean(epoch_times)]], fmt='o', label=label)
    axs[0].set_xticks(range(len(data)))
    axs[0].set_xticklabels(data.keys())
    axs[0].set_title('Epoch Training Time Intervals, 512 batch (min, mean, max)')
    axs[0].set_ylabel('Time (s)')
    axs[0].legend()

    # Graph 2: Test dataset processing times
    test_times = [values[1] for values in data.values()]
    axs[1].bar(data.keys(), test_times, color=['tab:blue', 'tab:orange', 'tab:green'])
    axs[1].set_title('Test Dataset (10k) Processing Time (batch image classification)')
    axs[1].set_ylabel('Time (ms)')

    # Graph 3: Accuracy over time
    for label, values in data.items():
        _, _, _, accuracy = values
        epochs = np.arange(1, len(accuracy) + 1)
        axs[2].plot(epochs, accuracy, label=f'{label} - Accuracy', marker='x')
    axs[2].set_title('Accuracy Over Time')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Accuracy (%)')
    axs[2].legend()

    # Graph 4: Loss over time
    for label, values in data.items():
        _, _, loss, _ = values
        epochs = np.arange(1, len(loss) + 1)
        axs[3].plot(epochs, loss, label=f'{label} - Loss', marker='o')
    axs[3].set_title('Loss Over Time')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Loss')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training and testing performance metrics.')
    parser.add_argument('--mlx', type=str, help='Path to the MLX log file')
    parser.add_argument('--torch_cpu', type=str, help='Path to the Torch CPU log file')
    parser.add_argument('--torch_gpu', type=str, help='Path to the Torch GPU log file')

    args = parser.parse_args()

    filenames = {'MLX': args.mlx, 'Torch CPU': args.torch_cpu, 'Torch GPU': args.torch_gpu}
    data = {}
    for label, filename in filenames.items():
        if filename:
            data[label] = extract_data(filename)

    plot_graphs(data)
