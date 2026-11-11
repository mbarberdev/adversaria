import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, images):
        # Inputs stay in [0, 1]; normalization happens inside the model.
        normalized = (images - 0.5) / 0.5
        return self.classifier(self.features(normalized))


def load_splits():
    dataset = CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=ToTensor(),
    )

    # Keep this split fixed, even when changing the model's training seed.
    train_data, val_data = random_split(
        dataset,
        [45000, 5000],
        generator=torch.Generator().manual_seed(42),
    )
    return dataset, train_data, val_data


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            scores = model(images)
            loss = loss_function(scores, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            total_correct += (scores.argmax(1) == labels).sum().item()
            total_examples += len(labels)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix GPU setup first.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Favor repeatability for this starter experiment.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    dataset, train_data, val_data = load_splits()
    val_indices = list(val_data.indices)

    if args.smoke:
        train_data = Subset(train_data, range(512))
        val_data = Subset(val_data, range(256))

    # Refuse to overwrite an earlier experiment.
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)

    configuration = {
        **vars(args),
        "split_seed": 42,
        "batch_size": 128,
        "learning_rate": 0.001,
        "train_examples": len(train_data),
        "validation_examples": len(val_data),
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / "config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    history = []
    best_accuracy = -1.0

    print(f"GPU: {configuration['gpu']}", flush=True)
    print(
        f"Training: {len(train_data)} images | Validation: {len(val_data)} images",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)

        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        )
        (output / "metrics.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_indices": val_indices,
            "classes": dataset.classes,
            "configuration": configuration,
        }
        torch.save(checkpoint, output / "latest.pt")

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(checkpoint, output / "best.pt")

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train accuracy {train_metrics['accuracy']:.1%} | "
            f"validation accuracy {val_metrics['accuracy']:.1%}",
            flush=True,
        )

    # Verify that saving and reloading preserves predictions.
    restored = SmallCNN().to(device)
    saved = torch.load(output / "latest.pt", map_location=device, weights_only=True)
    restored.load_state_dict(saved["model_state"])
    restored.eval()
    model.eval()
    images, _ = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        torch.testing.assert_close(model(images), restored(images))

    epochs = [entry["epoch"] for entry in history]
    plt.plot(
        epochs,
        [entry["train"]["accuracy"] for entry in history],
        label="Training",
    )
    plt.plot(
        epochs,
        [entry["validation"]["accuracy"] for entry in history],
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "accuracy.png")
    plt.close()

    print("Checkpoint reload verified.", flush=True)
    print(f"Results saved to: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
