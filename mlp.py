# Copyright © 2023 Apple Inc.

import argparse
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import mnist


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)


def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


def plot_training_curves(train_losses, train_accs, test_accs, save_dir='assets'):
    """Plot training loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(train_losses, marker='o', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot accuracy
    ax2.plot(train_accs, marker='o', label='Train Accuracy')
    ax2.plot(test_accs, marker='s', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to '{save_path}'")
    plt.show()


def plot_sample_predictions(model, images, labels, num_samples=16, save_dir='assets'):
    """Plot sample predictions with actual images."""
    os.makedirs(save_dir, exist_ok=True)

    # Get predictions for samples
    predictions = mx.argmax(model(images[:num_samples]), axis=1)

    # Create grid
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            # Reshape flattened image back to 28x28
            img = np.array(images[idx]).reshape(28, 28)
            true_label = int(labels[idx])
            pred_label = int(predictions[idx])

            ax.imshow(img, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}',
                        color=color, fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'sample_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved sample predictions to '{save_path}'")
    plt.show()


def plot_confusion_matrix(model, images, labels, num_classes=10, save_dir='assets'):
    """Plot confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)

    # Get all predictions
    predictions = mx.argmax(model(images), axis=1)
    predictions = np.array(predictions)
    labels_np = np.array(labels)

    # Build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels_np, predictions):
        cm[true, pred] += 1

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    im1 = ax1.imshow(cm, cmap='Blues')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xticks(range(num_classes))
    ax1.set_yticks(range(num_classes))
    plt.colorbar(im1, ax=ax1)

    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax1.text(j, i, cm[i, j],
                          ha="center", va="center", color="black", fontsize=8)

    # Normalized
    im2 = ax2.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xticks(range(num_classes))
    ax2.set_yticks(range(num_classes))
    plt.colorbar(im2, ax=ax2)

    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax2.text(j, i, f'{cm_normalized[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to '{save_path}'")
    plt.show()


def plot_misclassified_examples(model, images, labels, num_samples=16, save_dir='assets'):
    """Plot misclassified examples."""
    os.makedirs(save_dir, exist_ok=True)

    # Get predictions
    predictions = mx.argmax(model(images), axis=1)
    predictions = np.array(predictions)
    labels_np = np.array(labels)

    # Convert images to numpy for indexing
    images_np = np.array(images)

    # Find misclassified indices
    misclassified_idx = np.where(predictions != labels_np)[0]

    if len(misclassified_idx) == 0:
        print("No misclassified examples found!")
        return

    # Sample some misclassified examples
    sample_idx = misclassified_idx[:num_samples]

    # Create grid
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for idx, ax in enumerate(axes.flat):
        if idx < len(sample_idx):
            img_idx = sample_idx[idx]
            img = images_np[img_idx].reshape(28, 28)
            true_label = labels_np[img_idx]
            pred_label = predictions[img_idx]

            ax.imshow(img, cmap='gray')
            ax.set_title(f'True: {true_label}\nPred: {pred_label}',
                        color='red', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle(f'Misclassified Examples ({len(misclassified_idx)} total)',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'misclassified_examples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved misclassified examples to '{save_path}'")
    print(f"Total misclassified: {len(misclassified_idx)} out of {len(labels_np)}")
    plt.show()


def main(args):
    seed = 0
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    np.random.seed(seed)

    # Load the data
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, args.dataset)()
    )

    # Load the model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    optimizer = optim.SGD(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    @partial(mx.compile, inputs=model.state)
    def eval_fn(X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

    # Track metrics for visualization
    train_losses = []
    train_accs = []
    test_accs = []

    for e in range(num_epochs):
        tic = time.perf_counter()
        epoch_losses = []
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss = step(X, y)
            mx.eval(model.state)
            epoch_losses.append(loss.item())

        # Calculate metrics
        avg_loss = np.mean(epoch_losses)
        train_accuracy = eval_fn(train_images, train_labels)
        test_accuracy = eval_fn(test_images, test_labels)

        # Store for visualization
        train_losses.append(avg_loss)
        train_accs.append(train_accuracy.item())
        test_accs.append(test_accuracy.item())

        toc = time.perf_counter()
        print(
            f"Epoch {e}: Train Loss {avg_loss:.4f}, "
            f"Train Acc {train_accuracy.item():.3f}, "
            f"Test Acc {test_accuracy.item():.3f}, "
            f"Time {toc - tic:.3f} (s)"
        )

    # Create visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60 + "\n")

    plot_training_curves(train_losses, train_accs, test_accs)
    plot_sample_predictions(model, test_images, test_labels, num_samples=16)
    plot_confusion_matrix(model, test_images, test_labels, num_classes=num_classes)
    plot_misclassified_examples(model, test_images, test_labels, num_samples=16)

    print("\n" + "="*60)
    print(f"Final Test Accuracy: {test_accs[-1]:.3f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple MLP on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    args = parser.parse_args()

    main(args)
