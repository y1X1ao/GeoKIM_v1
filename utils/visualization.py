import matplotlib.pyplot as plt
import os

def plot_loss_curve(losses, save_path):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Training Loss Curve")
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def plot_feature_reconstruction(true_vals, pred_vals, feature_name, save_path):
    plt.figure()
    plt.scatter(true_vals, pred_vals, alpha=0.6)
    plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
    plt.xlabel(f"True {feature_name}")
    plt.ylabel(f"Reconstructed {feature_name}")
    plt.title(f"Reconstruction of {feature_name}")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()