import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_dir: Path, img_size: int, batch_size: int, valid_ratio: float, num_workers: int, seed: int):
    data_dir = data_dir.expanduser().resolve()
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Dataset expected at {data_dir} with subfolders train/ and test/."
        )

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transforms = valid_transforms

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    num_train = len(train_dataset)
    num_valid = int(valid_ratio * num_train)
    num_train -= num_valid

    train_dataset, valid_dataset = random_split(
        train_dataset,
        [num_train, num_valid],
        generator=torch.Generator().manual_seed(seed),
    )
    valid_dataset.dataset.transform = valid_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    classes = train_dataset.dataset.classes
    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader, classes


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_prob: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_prob),
        )
        # 4× MaxPool2d(2) sur 224×224 → 14×14
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class TransferResNet18(nn.Module):
    def __init__(self, num_classes: int = 2, feature_extract: bool = True, dropout_prob: float = 0.5):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def init_model(name: str, num_classes: int, feature_extract: bool = True):
    if name == 'scratch':
        return SimpleCNN(num_classes=num_classes)
    if name == 'resnet18':
        return TransferResNet18(num_classes=num_classes, feature_extract=feature_extract)
    raise ValueError(f"Unknown model name: {name}")


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float, momentum: float = 0.9):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def evaluate(model, dataloader, device, num_classes, max_batches=None):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    confusion = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    running_loss = 0.0
    num_samples = 0
    batch_count = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            confusion.update(preds, labels)
            num_samples += inputs.size(0)
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

    return {
        'loss': running_loss / num_samples,
        'accuracy': accuracy.compute().item(),
        'precision': precision.compute().item(),
        'recall': recall.compute().item(),
        'confusion': confusion.compute().cpu().numpy(),
    }


def train_one_epoch(model, dataloader, optimizer, device, max_batches=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    num_samples = 0
    batch_count = 0

    for inputs, labels in tqdm(dataloader, desc='Train', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    return running_loss / num_samples


def run_training(
    model_name,
    model,
    train_loader,
    valid_loader,
    test_loader,
    device,
    optimizer_name,
    lr,
    epochs,
    scheduler_step,
    scheduler_gamma,
    checkpoint_dir,
    max_train_batches=None,
    max_valid_batches=None,
    max_test_batches=None,
):
    model = model.to(device)
    optimizer = get_optimizer(model, optimizer_name, lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    best_val_loss = float('inf')
    best_checkpoint = checkpoint_dir / f'{model_name}_{optimizer_name}.pth'
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, max_batches=max_train_batches)
        valid_metrics = evaluate(model, valid_loader, device, num_classes=2, max_batches=max_valid_batches)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_metrics['loss'],
            'valid_accuracy': valid_metrics['accuracy'],
            'valid_precision': valid_metrics['precision'],
            'valid_recall': valid_metrics['recall'],
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={valid_metrics['loss']:.4f} "
            f"acc={valid_metrics['accuracy']:.4f} "
            f"prec={valid_metrics['precision']:.4f} "
            f"recall={valid_metrics['recall']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.5f}"
        )

        if valid_metrics['loss'] < best_val_loss:
            best_val_loss = valid_metrics['loss']
            torch.save(model.state_dict(), best_checkpoint)
            print(f"Saved checkpoint: {best_checkpoint}")

    print(f"Evaluating best model for {model_name}...")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_metrics = evaluate(model, test_loader, device, num_classes=2, max_batches=max_test_batches)
    print(
        f"Test | loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f}"
    )
    print('Confusion matrix:')
    print(test_metrics['confusion'])
    return history, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Cats vs Dogs training pipeline')
    parser.add_argument('--data-dir', default='data/cats_vs_dogs', help='Path to cats_vs_dogs dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler-step', type=int, default=3)
    parser.add_argument('--scheduler-gamma', type=float, default=0.5)
    parser.add_argument('--feature-extract', action='store_true', help='Freeze backbone for transfer learning')
    parser.add_argument('--max-train-batches', type=int, default=None, help='Limit training batches for quick debugging')
    parser.add_argument('--max-valid-batches', type=int, default=None, help='Limit validation batches for quick debugging')
    parser.add_argument('--max-test-batches', type=int, default=None, help='Limit test batches for quick debugging')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, test_loader, classes = get_data_loaders(
        Path(args.data_dir),
        args.img_size,
        args.batch_size,
        args.valid_ratio,
        num_workers=os.cpu_count() or 4,
        seed=args.seed,
    )

    print('Experiment A: CNN from scratch')
    model_a = init_model('scratch', num_classes=len(classes))
    run_training(
        'cnn_scratch',
        model_a,
        train_loader,
        valid_loader,
        test_loader,
        device,
        args.optimizer,
        args.lr,
        args.epochs,
        args.scheduler_step,
        args.scheduler_gamma,
        checkpoint_dir,
        max_train_batches=args.max_train_batches,
        max_valid_batches=args.max_valid_batches,
        max_test_batches=args.max_test_batches,
    )

    print('\nExperiment B: Transfer learning with ResNet18')
    model_b = init_model('resnet18', num_classes=len(classes), feature_extract=args.feature_extract)
    run_training(
        'resnet18_transfer',
        model_b,
        train_loader,
        valid_loader,
        test_loader,
        device,
        args.optimizer,
        args.lr,
        args.epochs,
        args.scheduler_step,
        args.scheduler_gamma,
        checkpoint_dir,
        max_train_batches=args.max_train_batches,
        max_valid_batches=args.max_valid_batches,
        max_test_batches=args.max_test_batches,
    )


if __name__ == '__main__':
    main()
