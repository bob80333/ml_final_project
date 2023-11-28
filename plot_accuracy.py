import torch
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("acc_file", help="Path to the accuracy file", default="checkpoints/accuracy.pt")
    args = parser.parse_args()
    # Load your accuracy tensor
    accuracy_tensor = torch.load(args.acc_file)

    # Filter out the -1 values and keep track of the corresponding indices/timesteps
    valid_accuracies = accuracy_tensor[accuracy_tensor != -1]
    valid_timesteps = torch.arange(len(accuracy_tensor))[accuracy_tensor != -1]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(valid_timesteps.numpy(), valid_accuracies.numpy(), marker='o')
    plt.title('Model Accuracy Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy.png')