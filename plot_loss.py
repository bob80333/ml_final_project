import torch
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("loss_file", help="Path to the loss file", default="checkpoints/loss.pt")
    args = parser.parse_args()

    # Load the loss tensor from file
    loss_tensor = torch.load(args.loss_file)

    # Make sure it's a 1-dimensional tensor
    if loss_tensor.ndim != 1:
        raise ValueError("The loss tensor should be 1-dimensional")

    # Convert to NumPy array for matplotlib if it's not already on CPU and in NumPy format
    loss_array = loss_tensor.cpu().numpy()

    # Plotting
    plt.figure(figsize=(10, 5))  # can adjust size if needed
    plt.plot(loss_array, label='Training Loss')
    plt.title('Training Loss over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')