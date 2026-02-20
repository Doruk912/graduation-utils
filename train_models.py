# ============================================================
# BIRD SPECIES CLASSIFICATION - V2
# ============================================================

import os
import time
import copy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ============================================================
# GPU CHECK
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM: {mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be extremely slow.")
    input("Press Enter to continue anyway, or Ctrl+C to abort...")


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # ==========================================
    # CHANGE THIS TO TRAIN DIFFERENT MODELS
    # Options: 'mobilenet_v2', 'resnet50', 'densenet121', 'inception_v3', 'vgg16'
    # ==========================================
    CURRENT_MODEL = 'mobilenet_v2'
    # ==========================================

    # Paths
    BASE_PATH = r"C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed"
    TRAIN_CSV = r"C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed\train.csv"
    VAL_CSV = r"C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed\val.csv"
    TEST_CSV = r"C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed\test.csv"
    SAVE_DIR = r"C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed\models"

    # CSV column names
    IMAGE_PATH_COL = 'image_path'
    LABEL_COL = 'label'
    SPECIES_COL = 'species'

    # Training parameters
    NUM_CLASSES = 23

    # Phase 1: Frozen backbone, train only classifier head
    FREEZE_EPOCHS = 5
    HEAD_LR = 1e-3

    # Phase 2: Unfreeze all layers, fine-tune with differential LR
    FINETUNE_EPOCHS = 20
    BACKBONE_LR = 1e-5
    FINETUNE_HEAD_LR = 1e-4

    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_MAX_NORM = 1.0

    # Batch size will be auto-set based on model (see below)
    BATCH_SIZE = 32

    # Early stopping (applied during fine-tuning phase)
    PATIENCE = 7
    MIN_DELTA = 0.001

    # Data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Random seed
    SEED = 42


# ============================================================
# PER-MODEL SETTINGS
# ============================================================

MODEL_SETTINGS = {
    'mobilenet_v2': {
        'batch_size': 32,
        'freeze_epochs': 5,
        'finetune_epochs': 20,
        'head_lr': 1e-3,
        'backbone_lr': 1e-5,
        'finetune_head_lr': 1e-4,
        'weight_decay': 1e-4,
    },
    'resnet50': {
        'batch_size': 16,
        'freeze_epochs': 5,
        'finetune_epochs': 20,
        'head_lr': 1e-3,
        'backbone_lr': 1e-5,
        'finetune_head_lr': 1e-4,
        'weight_decay': 1e-4,
    },
    'densenet121': {
        'batch_size': 16,
        'freeze_epochs': 5,
        'finetune_epochs': 20,
        'head_lr': 1e-3,
        'backbone_lr': 1e-5,
        'finetune_head_lr': 1e-4,
        'weight_decay': 1e-4,
    },
    'inception_v3': {
        'batch_size': 16,
        'freeze_epochs': 5,
        'finetune_epochs': 20,
        'head_lr': 1e-3,
        'backbone_lr': 1e-5,
        'finetune_head_lr': 1e-4,
        'weight_decay': 1e-4,
    },
    'vgg16': {
        'batch_size': 8,
        'freeze_epochs': 10,
        'finetune_epochs': 20,
        'head_lr': 1e-3,
        'backbone_lr': 1e-5,
        'finetune_head_lr': 1e-4,
        'weight_decay': 5e-4,
    },
}

# Apply per-model settings
settings = MODEL_SETTINGS.get(Config.CURRENT_MODEL, {})
Config.BATCH_SIZE = settings.get('batch_size', 16)
Config.FREEZE_EPOCHS = settings.get('freeze_epochs', 5)
Config.FINETUNE_EPOCHS = settings.get('finetune_epochs', 20)
Config.HEAD_LR = settings.get('head_lr', 1e-3)
Config.BACKBONE_LR = settings.get('backbone_lr', 1e-5)
Config.FINETUNE_HEAD_LR = settings.get('finetune_head_lr', 1e-4)
Config.WEIGHT_DECAY = settings.get('weight_decay', 1e-4)

# Set seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create save directory
os.makedirs(Config.SAVE_DIR, exist_ok=True)


# ============================================================
# VERIFY PATHS
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SETUP")
print("=" * 70)

paths_to_check = [
    ('Base path', Config.BASE_PATH),
    ('Train CSV', Config.TRAIN_CSV),
    ('Val CSV', Config.VAL_CSV),
    ('Test CSV', Config.TEST_CSV),
]

all_paths_ok = True
for name, path in paths_to_check:
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {name}: {path}")
    if not exists:
        all_paths_ok = False

if not all_paths_ok:
    print("\nERROR: Some paths don't exist!")
    exit(1)

print("\nAll paths verified!")


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURATION")
print("=" * 70)
print(f"  Model:              {Config.CURRENT_MODEL.upper()}")
print(f"  Batch size:         {Config.BATCH_SIZE}")
print(f"  Phase 1 (frozen):   {Config.FREEZE_EPOCHS} epochs, head LR={Config.HEAD_LR}")
print(f"  Phase 2 (finetune): {Config.FINETUNE_EPOCHS} epochs, backbone LR={Config.BACKBONE_LR}, head LR={Config.FINETUNE_HEAD_LR}")
print(f"  Weight decay:       {Config.WEIGHT_DECAY}")
print(f"  Grad clip norm:     {Config.GRAD_CLIP_MAX_NORM}")
print(f"  Early stop:         patience={Config.PATIENCE}")
print(f"  Workers:            {Config.NUM_WORKERS}")
print(f"  Save directory:     {Config.SAVE_DIR}")


# ============================================================
# DATA TRANSFORMS
# ============================================================

def get_transforms(model_name):
    """Get appropriate transforms for each model."""
    if model_name == 'inception_v3':
        img_size = 299
        resize_size = 342
    else:
        img_size = 224
        resize_size = 256

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ============================================================
# DATASET CLASS
# ============================================================

class BirdDataset(Dataset):
    """Custom Dataset for bird images."""

    def __init__(self, csv_file, image_path_col, label_col, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_path_col = image_path_col
        self.label_col = label_col
        self.transform = transform
        self.failed_count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx][self.image_path_col]
        label = int(self.data.iloc[idx][self.label_col])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            self.failed_count += 1
            if self.failed_count <= 5:
                print(f"  Warning: Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_model(model_name, num_classes, pretrained=True):
    """Get a pretrained model, freeze backbone, and replace the classifier head."""

    print(f"\n  Loading {model_name} with pretrained weights...")

    if model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)

        # Freeze all feature layers
        for param in model.features.parameters():
            param.requires_grad = False

        # Replace the classifier with a smaller, more appropriate one
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # Freeze all layers except the final FC
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'inception_v3':
        model = models.inception_v3(weights='IMAGENET1K_V1' if pretrained else None)

        # Freeze all layers except the classifier heads
        for param in model.parameters():
            param.requires_grad = False

        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)

        # Freeze all feature layers
        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)

        # Freeze all feature layers
        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def unfreeze_model(model, model_name):
    """Unfreeze all layers for fine-tuning phase."""
    for param in model.parameters():
        param.requires_grad = True
    print(f"  All layers unfrozen for fine-tuning.")


def get_parameter_groups(model, model_name, backbone_lr, head_lr, weight_decay):
    """Create parameter groups with differential learning rates."""

    backbone_params = []
    head_params = []

    if model_name == 'vgg16':
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

    elif model_name == 'resnet50':
        head_params = list(model.fc.parameters())
        backbone_params = [p for name, p in model.named_parameters()
                           if not name.startswith('fc')]

    elif model_name == 'inception_v3':
        head_params = list(model.fc.parameters()) + list(model.AuxLogits.fc.parameters())
        backbone_params = [p for name, p in model.named_parameters()
                           if not name.startswith('fc') and 'AuxLogits.fc' not in name]

    elif model_name == 'mobilenet_v2':
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

    elif model_name == 'densenet121':
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

    param_groups = [
        {'params': [p for p in backbone_params if p.requires_grad],
         'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': [p for p in head_params if p.requires_grad],
         'lr': head_lr, 'weight_decay': weight_decay},
    ]

    return param_groups


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"    Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


# ============================================================
# TRAINING FUNCTION (single phase)
# ============================================================

def train_phase(model, model_name, train_loader, val_loader, criterion,
                optimizer, scheduler, num_epochs, device, save_dir,
                phase_name, history, best_acc, best_epoch, start_time,
                early_stopping=None):
    """Train model for one phase and return updated history."""

    print(f"\n{'=' * 70}")
    print(f"  {phase_name}: {model_name.upper()}")
    print(f"{'=' * 70}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        epoch_start = time.time()
        global_epoch = len(history['train_loss']) + 1

        print(f"\n  --- {phase_name} Epoch {epoch + 1}/{num_epochs} (Global: {global_epoch}) ---")

        # ========== TRAINING PHASE ==========
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        train_pbar = tqdm(train_loader, desc="    Train",
                          bar_format='{l_bar}{bar:30}{r_bar}')

        for inputs, labels in train_pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if model_name == 'inception_v3' and model.training:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_MAX_NORM)

            optimizer.step()

            _, preds = torch.max(outputs, 1)
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += batch_size

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{running_corrects / total_samples:.3f}'
            })

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects / len(train_loader.dataset)

        # ========== VALIDATION PHASE ==========
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        val_pbar = tqdm(val_loader, desc="    Val  ",
                        bar_format='{l_bar}{bar:30}{r_bar}')

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects / len(val_loader.dataset)

        # Step the scheduler
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)

        # Get current LR (from head params group)
        current_lr = optimizer.param_groups[-1]['lr']
        backbone_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 1 else current_lr

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time

        print(f"    Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"    Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")
        print(f"    Head LR: {current_lr:.6f} | Backbone LR: {backbone_lr:.6f} | Time: {epoch_time / 60:.1f}m | Total: {elapsed / 60:.1f}m")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_epoch = global_epoch
            best_model_wts = copy.deepcopy(model.state_dict())

            checkpoint_path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, checkpoint_path)
            print(f"    >>> NEW BEST! Saved (acc: {best_acc:.4f})")

        if early_stopping is not None:
            if early_stopping(epoch_val_loss):
                print(f"\n  EARLY STOPPING at {phase_name} epoch {epoch + 1}")
                print(f"  Best accuracy: {best_acc:.4f} at global epoch {best_epoch}")
                break

    # Restore best weights
    model.load_state_dict(best_model_wts)

    return model, history, best_acc, best_epoch


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""

    print(f"\n  Evaluating on test set...")

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="    Test "):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score,
                                 top_k_accuracy_score)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5)

    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_training_history(history, model_name, save_dir, freeze_epochs):
    """Plot and save training history with phase boundary."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    if freeze_epochs < len(history['train_loss']):
        axes[0].axvline(x=freeze_epochs + 0.5, color='green', linestyle='--',
                        alpha=0.7, label='Unfreeze point')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name.upper()} - Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    if freeze_epochs < len(history['train_loss']):
        axes[1].axvline(x=freeze_epochs + 0.5, color='green', linestyle='--',
                        alpha=0.7, label='Unfreeze point')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name.upper()} - Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training history saved: {save_path}")


def plot_confusion_matrix(labels, predictions, class_names, model_name, save_dir):
    """Plot and save confusion matrix."""

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
    plt.colorbar(label='Proportion')

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=7)
    plt.yticks(tick_marks, class_names, fontsize=7)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    model_name = Config.CURRENT_MODEL

    print("\n" + "#" * 70)
    print(f"#  BIRD SPECIES CLASSIFICATION: {model_name.upper()}")
    print("#" * 70)

    # Load label mapping to get class names in order
    label_mapping_path = os.path.join(Config.BASE_PATH, 'label_mapping.csv')
    label_map_df = pd.read_csv(label_mapping_path)
    class_names = label_map_df.sort_values('label')['species'].tolist()
    print(f"\n  Number of classes: {len(class_names)}")

    # Verify a few image paths
    train_df = pd.read_csv(Config.TRAIN_CSV)
    sample_paths = train_df[Config.IMAGE_PATH_COL].sample(3, random_state=42).tolist()
    print(f"\n  Quick path check:")
    for p in sample_paths:
        exists = os.path.exists(p)
        short = "..." + p[-50:] if len(p) > 50 else p
        print(f"    {'OK' if exists else 'MISSING'}: {short}")
        if not exists:
            print("\n  ERROR: Image paths don't exist! Check your CSV.")
            exit(1)

    # Get transforms
    train_transform, val_transform = get_transforms(model_name)

    # Create datasets
    print(f"\n  Creating datasets...")
    train_dataset = BirdDataset(
        Config.TRAIN_CSV, Config.IMAGE_PATH_COL, Config.LABEL_COL,
        transform=train_transform
    )
    val_dataset = BirdDataset(
        Config.VAL_CSV, Config.IMAGE_PATH_COL, Config.LABEL_COL,
        transform=val_transform
    )
    test_dataset = BirdDataset(
        Config.TEST_CSV, Config.IMAGE_PATH_COL, Config.LABEL_COL,
        transform=val_transform
    )

    print(f"    Train: {len(train_dataset):,} images")
    print(f"    Val:   {len(val_dataset):,} images")
    print(f"    Test:  {len(test_dataset):,} images")

    # Create data loaders
    use_persistent = Config.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=use_persistent
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=use_persistent
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=use_persistent
    )

    # Create model (backbone frozen by default)
    model = get_model(model_name, Config.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (head only)")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # PHASE 1: Train only the classifier head (backbone frozen)
    # ========================================================

    print("\n" + "=" * 70)
    print("  PHASE 1: TRAINING CLASSIFIER HEAD (backbone frozen)")
    print("=" * 70)

    # Only optimize trainable (head) parameters
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_phase1 = optim.Adam(head_params, lr=Config.HEAD_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase1, T_max=Config.FREEZE_EPOCHS, eta_min=Config.HEAD_LR * 0.1
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    start_time = time.time()

    model, history, best_acc, best_epoch = train_phase(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_phase1,
        scheduler=scheduler_phase1,
        num_epochs=Config.FREEZE_EPOCHS,
        device=device,
        save_dir=Config.SAVE_DIR,
        phase_name="PHASE 1 (Frozen)",
        history=history,
        best_acc=0.0,
        best_epoch=0,
        start_time=start_time,
        early_stopping=None  # No early stopping in phase 1
    )

    phase1_time = time.time() - start_time
    print(f"\n  Phase 1 complete. Best val acc: {best_acc:.4f} at epoch {best_epoch}")
    print(f"  Phase 1 time: {phase1_time / 60:.1f} minutes")

    # ========================================================
    # PHASE 2: Fine-tune all layers with differential LR
    # ========================================================

    print("\n" + "=" * 70)
    print("  PHASE 2: FINE-TUNING ALL LAYERS (differential learning rates)")
    print("=" * 70)

    # Unfreeze all layers
    unfreeze_model(model, model_name)

    total_params2, trainable_params2 = count_parameters(model)
    print(f"  Trainable parameters: {trainable_params2:,} (all layers)")

    # Create optimizer with differential learning rates
    param_groups = get_parameter_groups(
        model, model_name,
        backbone_lr=Config.BACKBONE_LR,
        head_lr=Config.FINETUNE_HEAD_LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    optimizer_phase2 = optim.Adam(param_groups)
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=Config.FINETUNE_EPOCHS,
        eta_min=Config.BACKBONE_LR * 0.1
    )

    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA
    )

    model, history, best_acc, best_epoch = train_phase(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_phase2,
        scheduler=scheduler_phase2,
        num_epochs=Config.FINETUNE_EPOCHS,
        device=device,
        save_dir=Config.SAVE_DIR,
        phase_name="PHASE 2 (Fine-tune)",
        history=history,
        best_acc=best_acc,
        best_epoch=best_epoch,
        start_time=start_time,
        early_stopping=early_stopping
    )

    total_time = time.time() - start_time

    # Save final model
    final_path = os.path.join(Config.SAVE_DIR, f'{model_name}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'history': history,
        'training_time': total_time
    }, final_path)

    print(f"\n  Training complete for {model_name.upper()}")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Best epoch: {best_epoch}, Best val acc: {best_acc:.4f}")

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device)

    # Print final results
    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS: {model_name.upper()}")
    print("=" * 70)
    print(f"  Validation Accuracy:  {best_acc:.4f} ({best_acc * 100:.2f}%)")
    print(f"  Test Accuracy:        {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)")
    print(f"  Top-5 Accuracy:       {test_metrics['top5_accuracy']:.4f} ({test_metrics['top5_accuracy'] * 100:.2f}%)")
    print(f"  Precision:            {test_metrics['precision']:.4f}")
    print(f"  Recall:               {test_metrics['recall']:.4f}")
    print(f"  F1 Score:             {test_metrics['f1_score']:.4f}")
    print(f"  Training Time:        {total_time / 60:.1f} minutes")

    # Plot
    plot_training_history(history, model_name, Config.SAVE_DIR, Config.FREEZE_EPOCHS)
    plot_confusion_matrix(
        test_metrics['labels'],
        test_metrics['predictions'],
        class_names,
        model_name,
        Config.SAVE_DIR
    )

    # Save results to comparison CSV
    results = {
        'model': model_name,
        'total_parameters': total_params2,
        'trainable_parameters': trainable_params2,
        'batch_size': Config.BATCH_SIZE,
        'freeze_epochs': Config.FREEZE_EPOCHS,
        'finetune_epochs': len(history['train_loss']) - Config.FREEZE_EPOCHS,
        'total_epochs': len(history['train_loss']),
        'head_lr': Config.HEAD_LR,
        'backbone_lr': Config.BACKBONE_LR,
        'val_accuracy': best_acc,
        'test_accuracy': test_metrics['accuracy'],
        'top5_accuracy': test_metrics['top5_accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1_score': test_metrics['f1_score'],
        'training_time_min': total_time / 60,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    results_csv_path = os.path.join(Config.SAVE_DIR, 'model_comparison.csv')
    try:
        if os.path.exists(results_csv_path):
            existing = pd.read_csv(results_csv_path)
            existing = existing[existing['model'] != model_name]
            new_df = pd.concat([existing, pd.DataFrame([results])], ignore_index=True)
        else:
            new_df = pd.DataFrame([results])

        new_df.to_csv(results_csv_path, index=False)
        print(f"\n  Results saved to: {results_csv_path}")
    except PermissionError:
        # Fallback: save to a different file if the original is locked
        alt_path = os.path.join(Config.SAVE_DIR, f'model_comparison_{model_name}.csv')
        pd.DataFrame([results]).to_csv(alt_path, index=False)
        print(f"\n  WARNING: Could not write to {results_csv_path} (file locked)")
        print(f"  Results saved to: {alt_path}")

    # Cleanup
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  DONE!")
    print("=" * 70)
    remaining = [m for m in ['mobilenet_v2', 'resnet50', 'densenet121',
                              'inception_v3', 'vgg16'] if m != model_name]
    print(f"\n  To train the next model:")
    print(f"  1. Change CURRENT_MODEL to one of: {remaining}")
    print(f"  2. Run this script again\n")

    return model, history, test_metrics


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model, history, metrics = main()