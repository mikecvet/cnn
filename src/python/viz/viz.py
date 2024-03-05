import matplotlib.pyplot as plt
import argparse
import re

def load_data_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Extract loss and accuracy
    loss_match = re.search(r'loss: \[([^\]]+)\]', content)
    accuracy_match = re.search(r'accuracy: \[([^\]]+)\]', content)

    loss = [float(x) for x in loss_match.group(1).split(', ')] if loss_match else []
    accuracy = [float(x) for x in accuracy_match.group(1).split(', ')] if accuracy_match else []

    return loss, accuracy

def plot_data(data):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')

    for label, (loss, accuracy) in data.items():
        epochs = range(1, len(loss) + 1)
        ax1.plot(epochs, loss, label=f'Loss - {label}', color='tab:red', marker='o')
        ax2.plot(epochs, accuracy, label=f'Accuracy - {label}', color='tab:blue', linestyle='--', marker='x')

    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Model Performance: Loss and Accuracy Across Epochs')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize benchmark logs with loss and accuracy.')
    parser.add_argument('--mlx', type=str, help='Path to the MLX log file')
    parser.add_argument('--torch_cpu', type=str, help='Path to the Torch CPU log file')
    parser.add_argument('--torch_gpu', type=str, help='Path to the Torch GPU log file')

    args = parser.parse_args()

    data = {}
    if args.mlx:
        data['MLX'] = load_data_from_file(args.mlx)
    if args.torch_cpu:
        data['Torch CPU'] = load_data_from_file(args.torch_cpu)
    if args.torch_gpu:
        data['Torch GPU'] = load_data_from_file(args.torch_gpu)

    plot_data(data)
