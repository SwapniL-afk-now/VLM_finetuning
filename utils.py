# utils.py
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_metrics(metrics, output_dir, total_steps):
    """Plots the training loss and learning rate."""
    if not metrics["loss"] or not metrics["lr"]:
        print("No metrics to plot.")
        return

    # Generate step numbers for plotting based on the number of logged points
    num_logged_steps = len(metrics["loss"])
    # Assume logging happens every `logging_steps` after an optimizer step
    # This requires knowing logging_steps, but we don't have config here easily.
    # A simpler approach is just to plot against the logged step count.
    steps_axis = list(range(num_logged_steps)) # Simple index for x-axis

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(steps_axis, metrics["loss"], label="Training Loss")
    plt.title("Training Loss vs Logged Steps")
    plt.xlabel("Logged Step Index")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Smooth the loss curve for better visualization (optional)
    # window_size = 50 # Adjust window size as needed
    # if len(metrics["loss"]) >= window_size:
    #     smoothed_loss = np.convolve(metrics["loss"], np.ones(window_size)/window_size, mode='valid')
    #     smoothed_steps = steps_axis[window_size-1:]
    #     plt.plot(smoothed_steps, smoothed_loss, label=f"Smoothed Loss (window={window_size})", alpha=0.7)
    #     plt.legend()


    # Plot Learning Rate
    plt.subplot(1, 2, 2)
    plt.plot(steps_axis, metrics["lr"], label="Learning Rate", color="orange")
    plt.title("Learning Rate vs Logged Steps")
    plt.xlabel("Logged Step Index")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_metrics.png")
    try:
        plt.savefig(plot_path)
        print(f"Training metrics plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Comment out if running in a non-interactive environment
    plt.close() # Close the figure to free memory