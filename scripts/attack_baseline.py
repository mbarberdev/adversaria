# this is a FGSM one step attack that uses image gradient to choose pixel changes
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from train_baseline import SmallCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["configuration"]["smoke"]:
        raise ValueError("Use a full baseline checkpoint, not a smoke run.")

    model = SmallCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = CIFAR10(root="data", train=True, download=False, transform=ToTensor())
    # Reuse the exact held-out validation images from training.
    validation = Subset(dataset, checkpoint["validation_indices"])
    loader = DataLoader(validation, batch_size=128, shuffle=False, num_workers=0)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    loss_function = nn.CrossEntropyLoss()

    for epsilon in (0.0, 2 / 255, 4 / 255, 8 / 255):
        total = 0
        clean_correct = 0
        attacked_correct = 0
        newly_wrong = 0

        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device).requires_grad_(True)
            labels = labels.to(device)

            scores = model(images)
            loss = loss_function(scores, labels)
            gradient = torch.autograd.grad(loss, images)[0]

            attacked = (images.detach() + epsilon * gradient.sign()).clamp(0, 1)

            assert (attacked - images.detach()).abs().max().item() <= (epsilon + 1e-6)

            with torch.no_grad():
                clean_predictions = scores.argmax(1)
                attacked_predictions = model(attacked).argmax(1)

            if epsilon == 0:
                assert torch.equal(clean_predictions, attacked_predictions)

            originally_correct = clean_predictions == labels
            now_correct = attacked_predictions == labels

            total += len(labels)
            clean_correct += originally_correct.sum().item()
            attacked_correct += now_correct.sum().item()
            newly_wrong += (originally_correct & ~now_correct).sum().item()

            if batch_index == 0 and epsilon == 8 / 255:
                # Top row: originals. Bottom row: attacked versions.
                gallery = torch.cat([images[:8].detach().cpu(), attacked[:8].cpu()])
                save_image(gallery, output / "comparison.png", nrow=8)
                (output / "examples.json").write_text(
                    json.dumps(
                        {
                            "true_labels": labels[:8].tolist(),
                            "clean_predictions": clean_predictions[:8].tolist(),
                            "attacked_predictions": attacked_predictions[:8].tolist(),
                            "classes": dataset.classes,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        result = {
            "epsilon": epsilon,
            "examples": total,
            "clean_accuracy": clean_correct / total,
            "attacked_accuracy": attacked_correct / total,
            "success_rate_on_initially_correct": (
                newly_wrong / clean_correct if clean_correct else None
            ),
        }
        results.append(result)
        print(
            f"epsilon={epsilon:.5f} | "
            f"clean={result['clean_accuracy']:.1%} | "
            f"attacked={result['attacked_accuracy']:.1%}",
            flush=True,
        )

    (output / "results.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "attack": "untargeted FGSM",
                "partition": "validation",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
