import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fig_4_4_1_loss_curves.png")

def generate_plot():
    # Mock data representing 10 epochs
    epochs = list(range(1, 11))
    
    # Training loss (decreases steadily)
    train_loss = [0.65, 0.45, 0.32, 0.22, 0.15, 0.11, 0.08, 0.06, 0.04, 0.0342]
    
    # Validation loss (decreases, then plateaus/slightly overfits after epoch 7)
    val_loss = [0.68, 0.49, 0.38, 0.30, 0.24, 0.19, 0.16, 0.14, 0.13, 0.125]
    
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(epochs, train_loss, 'b-', marker='o', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', marker='s', linewidth=2, label='Validation Loss')
    
    # Highlight convergence point
    plt.axvline(x=7, color='gray', linestyle='--', alpha=0.7)
    plt.text(7.2, 0.5, 'Convergence Point\n(Epoch 7)', color='gray', fontsize=11)
    
    # Add titles and labels
    plt.title('Training and Validation Loss Curves', fontsize=16, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
    plt.xticks(epochs)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved successfully to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_plot()
