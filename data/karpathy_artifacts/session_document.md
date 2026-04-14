# Autoresearch-Lite: Iterative CIFAR-10 CNN Optimization

## Abstract

An autonomous ML research agent iteratively optimized a CNN for CIFAR-10 classification by making one focused change per experiment. Starting from a baseline of 70.94% validation accuracy, the agent ran 20 experiments over hyperparameters (learning rate, optimizer, weight decay, epochs, batch size), architecture (filters, layers, activations, residual connections), and data augmentation (flips, crops, color jitter). The best configuration achieved 73.99% accuracy. Two experiments crashed. Changes were kept only if they improved accuracy; otherwise the model reverted to the previous best.

## Research Protocol (Agent Instructions)

# Autoresearch-Lite: Research Org Code

You are an autonomous ML researcher. Your goal is to **maximize val_accuracy** on CIFAR-10
by iteratively modifying `train.py`.

## Rules

1. You may ONLY modify `train.py`. No other files.
2. The TIME_BUDGET (60 seconds) and output format are IMMUTABLE. Do not change them.
3. Each experiment must be a single, focused change. Do not make multiple unrelated changes at once.
4. After each experiment, you will receive the results. If val_accuracy improved, the change is KEPT. Otherwise, it is DISCARDED and train.py is reverted.
5. Your goal is to maximize val_accuracy. Secondary goals: reduce memory usage, reduce training time.

## What you can modify in train.py

### Hyperparameters (at the top of the file)
- BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS
- OPTIMIZER ("sgd", "adam", "adamw")
- LR_SCHEDULER ("cosine", "step", "none")
- DROPOUT
- NUM_FILTERS_1, NUM_FILTERS_2, NUM_FILTERS_3, FC_SIZE
- USE_BATCHNORM, ACTIVATION ("relu", "gelu", "silu")
- USE_HORIZONTAL_FLIP, USE_RANDOM_CROP, USE_COLOR_JITTER

### Architecture (the SimpleCNN class)
- Add residual connections
- Add more layers
- Change pooling strategy
- Add attention mechanisms
- Any architectural change that fits in 60 seconds

### Training loop
- Learning rate warmup
- Gradient clipping
- Mixed precision training
- Label smoothing
- Any training technique

## Strategy tips

- Start with optimizer and LR tuning (biggest impact, lowest risk)
- Then try architectural changes (residual connections, more depth)
- Data augmentation changes are safe experiments
- Be careful with changes that increase memory/time (may exceed TIME_BUDGET)
- If something crashes, understand why before trying again

## Output

For each experiment, provide:
1. A short description of what you changed and why (one sentence)
2. The modified train.py code


## Experiment Results

Each row represents one experiment. Status: keep = improved over previous best, discard = did not improve, crash = runtime error.

| commit | val_accuracy | memory_gb | status | description |
| --- | --- | --- | --- | --- |
| 2108755 | 0.709400 | 1.1 | keep | baseline |
| 61b4eb4 | 0.693100 | 1.1 | discard | Increase learning rate from 0.01 to 0.1 to potentially achieve faster convergence and better accuracy with SGD optimizer. |
| 1a88964 | 0.444700 | 1.1 | discard | Switch optimizer from SGD to AdamW which often provides better convergence and generalization on CIFAR-10. |
| d0545bb | 0.705000 | 1.1 | discard | Reduce learning rate to 0.005 to provide more stable convergence with SGD and cosine annealing scheduler. |
| 666e157 | 0.695200 | 1.1 | discard | Add dropout of 0.3 to regularize the model and potentially improve generalization performance. |
| 98aea59 | 0.736300 | 1.1 | keep | Increase the number of epochs from 10 to 15 to allow the model more training time which should improve convergence and validation accuracy. |
| 437c019 | 0.730400 | 1.1 | discard | Switch from ReLU to GELU activation function which often provides better performance on image classification tasks. |
| 72ce38e | 0.000000 | 0.0 | crash | Add residual connections to the CNN architecture to improve gradient flow and enable better feature learning. |
| 7e11bde | 0.730900 | 1.1 | discard | Increase the model capacity by doubling the number of filters in each convolutional layer (64, 128, 256) to improve feature representation capability. |
| f83873e | 0.712000 | 1.1 | discard | Enable color jitter data augmentation to improve model generalization through additional training data variety. |
| 1261844 | 0.734600 | 1.1 | discard | Increase batch size from 128 to 256 to improve training stability and potentially better convergence while staying within the time budget. |
| 957e801 | 0.729400 | 1.1 | discard | Increase the FC hidden layer size from 256 to 512 to improve the model's capacity for learning complex feature representations. |
| 8d7d7c1 | 0.712500 | 1.1 | discard | Add a fourth convolutional block with 256 filters to increase model depth and feature representation capacity without significantly impacting training time. |
| 8f0cc3f | 0.730900 | 1.1 | discard | Increase learning rate to 0.015 to achieve faster convergence while maintaining the stable SGD+cosine annealing setup that achieved the current best accuracy. |
| d1b77b9 | 0.682300 | 1.1 | discard | Add gradient clipping with max norm 1.0 to improve training stability and potentially achieve better convergence. |
| e7e8ecb | 0.719900 | 1.1 | discard | Increase learning rate to 0.012 for slightly faster convergence while staying closer to the stable 0.01 baseline that achieved good results. |
| c7b5349 | 0.687900 | 1.1 | discard | Switch from cosine annealing to step scheduler with step size of 5 epochs to potentially improve learning rate decay timing for the 15-epoch training. |
| d13e3e8 | 0.715800 | 1.1 | discard | Increase learning rate to 0.013 to find the optimal point between the current best 0.01 and the too-high 0.015 that was tried earlier. |
| 0000000 | 0.000000 | 0.0 | crash | LLM error: Expecting value: line 1 column 1 (char 0) |
| 44fb21c | 0.739900 | 1.1 | keep | Reduce weight decay from 1e-4 to 5e-5 to reduce regularization strength and potentially allow the model to fit the training data better while maintaining generalization. |
| febd989 | 0.723000 | 1.1 | discard | Reduce weight decay further from 5e-5 to 2e-5 to continue the trend of reducing regularization strength that improved accuracy in the previous experiment. |

## Summary Statistics

- **Total experiments**: 21
- **Kept (improved)**: 3
- **Discarded**: 16
- **Crashed**: 2
- **Best accuracy**: 0.7399
- **Worst accuracy (non-crash)**: 0.4447
- **Mean accuracy (non-crash)**: 0.7017

## Improvement Trajectory (Kept Experiments)

1. **0.7094** (2108755): baseline
1. **0.7363** (98aea59): Increase the number of epochs from 10 to 15 to allow the model more training time which should improve convergence and validation accuracy.
1. **0.7399** (44fb21c): Reduce weight decay from 1e-4 to 5e-5 to reduce regularization strength and potentially allow the model to fit the training data better while maintaining generalization.

## Crash Analysis

- **72ce38e**: Add residual connections to the CNN architecture to improve gradient flow and enable better feature learning.
- **0000000**: LLM error: Expecting value: line 1 column 1 (char 0)

## Git History

```
44fb21c Reduce weight decay from 1e-4 to 5e-5 to reduce regularization strength
98aea59 Increase the number of epochs from 10 to 15 to allow the model more trai
2108755 fix: use existing commit hash for baseline instead of re-committing
c31a9f1 initial setup: CIFAR-10 CNN autoresearch-lite
```

## Final Model Code (train.py)

This is the best configuration after all experiments:

```python
"""CIFAR-10 CNN training script for autoresearch-lite.
This file is the ONLY file the LLM agent modifies.

Output format (printed to stdout at the end):
    val_accuracy=0.7530
    training_seconds=28.4
    total_seconds=35.1
    peak_memory_mb=1024.5
    num_params_K=156.2
    num_epochs=10
"""

import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ============================================================
# HYPERPARAMETERS — The agent modifies these
# ============================================================
BATCH_SIZE = 128
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-5
NUM_EPOCHS = 15
OPTIMIZER = "sgd"  # "sgd", "adam", "adamw"
LR_SCHEDULER = "cosine"  # "cosine", "step", "none"
DROPOUT = 0.0

# Architecture
NUM_FILTERS_1 = 32
NUM_FILTERS_2 = 64
NUM_FILTERS_3 = 128
FC_SIZE = 256
USE_BATCHNORM = True
ACTIVATION = "relu"  # "relu", "gelu", "silu"

# Data augmentation
USE_HORIZONTAL_FLIP = True
USE_RANDOM_CROP = True
USE_COLOR_JITTER = False

# Time budget (seconds) — DO NOT MODIFY
TIME_BUDGET = 60


# ============================================================
# MODEL
# ============================================================
def get_activation():
    if ACTIVATION == "relu":
        return nn.ReLU(inplace=True)
    elif ACTIVATION == "gelu":
        return nn.GELU()
    elif ACTIVATION == "silu":
        return nn.SiLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers1 = [nn.Conv2d(3, NUM_FILTERS_1, 3, padding=1)]
        if USE_BATCHNORM:
            layers1.append(nn.BatchNorm2d(NUM_FILTERS_1))
        layers1.append(get_activation())
        layers1.append(nn.MaxPool2d(2, 2))
        self.block1 = nn.Sequential(*layers1)

        layers2 = [nn.Conv2d(NUM_FILTERS_1, NUM_FILTERS_2, 3, padding=1)]
        if USE_BATCHNORM:
            layers2.append(nn.BatchNorm2d(NUM_FILTERS_2))
        layers2.append(get_activation())
        layers2.append(nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(*layers2)

        layers3 = [nn.Conv2d(NUM_FILTERS_2, NUM_FILTERS_3, 3, padding=1)]
        if USE_BATCHNORM:
            layers3.append(nn.BatchNorm2d(NUM_FILTERS_3))
        layers3.append(get_activation())
        layers3.append(nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(*layers3)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(NUM_FILTERS_3 * 4 * 4, FC_SIZE),
            get_activation(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_SIZE, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ============================================================
# DATA
# ============================================================
def get_transforms():
    train_transforms = []
    if USE_RANDOM_CROP:
        train_transforms.append(transforms.RandomCrop(32, padding=4))
    if USE_HORIZONTAL_FLIP:
        train_transforms.append(transforms.RandomHorizontalFlip())
    if USE_COLOR_JITTER:
        train_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


# ============================================================
# TRAINING
# ============================================================
def main():
    total_start = time.time()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data
    train_transform, test_transform = get_transforms()
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    pin = device.type == "cuda"
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=pin
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=pin
    )

    # Model
    model = SimpleCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1000.0

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
        )

    # Scheduler
    if LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif LR_SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, NUM_EPOCHS // 3), gamma=0.1)
    else:
        scheduler = None

    # Train
    train_start = time.time()
    epochs_completed = 0
    for epoch in range(NUM_EPOCHS):
        elapsed = time.time() - total_start
        if elapsed > TIME_BUDGET:
            break

        model.train()
        for inputs, targets in trainloader:
            if time.time() - total_start > TIME_BUDGET:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        epochs_completed += 1

    training_seconds = time.time() - train_start

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_accuracy = correct / total
    total_seconds = time.time() - total_start

    # Memory
    if device.type == "mps":
        peak_memory_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
    elif device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory_mb = 0.0

    # Output — DO NOT MODIFY THIS FORMAT
    print(f"val_accuracy={val_accuracy:.6f}")
    print(f"training_seconds={training_seconds:.1f}")
    print(f"total_seconds={total_seconds:.1f}")
    print(f"peak_memory_mb={peak_memory_mb:.1f}")
    print(f"num_params_K={num_params:.1f}")
    print(f"num_epochs={epochs_completed}")


if __name__ == "__main__":
    main()

```

## Best Model Hyperparameters

Extracted from the final train.py:

| Parameter | Value |
|-----------|-------|
| BATCH_SIZE | 128 |
| LEARNING_RATE | 0.01 |
| WEIGHT_DECAY | 5e-5 |
| NUM_EPOCHS | 15 |
| OPTIMIZER | "sgd" |
| LR_SCHEDULER | "cosine" |
| DROPOUT | 0.0 |
| NUM_FILTERS_1 | 32 |
| NUM_FILTERS_2 | 64 |
| NUM_FILTERS_3 | 128 |
| FC_SIZE | 256 |
| USE_BATCHNORM | True |
| ACTIVATION | "relu" |
| USE_HORIZONTAL_FLIP | True |
| USE_RANDOM_CROP | True |
| USE_COLOR_JITTER | False |
