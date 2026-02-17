# ============================================================
# BIRD SPECIES CLASSIFICATION
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
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # Batch size will be auto-set based on model (see below)
    BATCH_SIZE = 32

    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.001

    # Data loading
    # If you get multiprocessing errors, change NUM_WORKERS to 0
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Random seed
    SEED = 42


# Auto-select batch size based on model (for RTX 4050 6GB VRAM)
RECOMMENDED_BATCH_SIZES = {
    'mobilenet_v2': 32,
    'resnet50': 16,
    'densenet121': 16,
    'inception_v3': 16,
    'vgg16': 8,
}
Config.BATCH_SIZE = RECOMMENDED_BATCH_SIZES.get(Config.CURRENT_MODEL, 16)


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
print(f"  Model:            {Config.CURRENT_MODEL.upper()}")
print(f"  Batch size:       {Config.BATCH_SIZE}")
print(f"  Max epochs:       {Config.NUM_EPOCHS}")
print(f"  Learning rate:    {Config.LEARNING_RATE}")
print(f"  Early stop:       patience={Config.PATIENCE}")
print(f"  Workers:          {Config.NUM_WORKERS}")
print(f"  Save directory:   {Config.SAVE_DIR}")


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
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
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
    """Get a pretrained model and modify the final layer."""

    print(f"\n  Loading {model_name} with pretrained weights...")

    if model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'inception_v3':
        model = models.inception_v3(weights='IMAGENET1K_V1' if pretrained else None)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
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
# TRAINING FUNCTION
# ============================================================

def train_model(model, model_name, train_loader, val_loader, criterion,
                optimizer, scheduler, num_epochs, device, save_dir):
    """Train a single model and return training history."""

    print(f"\n{'=' * 70}")
    print(f"  TRAINING: {model_name.upper()}")
    print(f"{'=' * 70}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        print(f"\n  --- Epoch {epoch + 1}/{num_epochs} ---")

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

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time

        print(f"    Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"    Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")
        print(f"    LR: {current_lr:.6f} | Time: {epoch_time / 60:.1f}m | Total: {elapsed / 60:.1f}m")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())

            checkpoint_path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, checkpoint_path)
            print(f"    >>> NEW BEST! Saved (acc: {best_acc:.4f})")

        if early_stopping(epoch_val_loss):
            print(f"\n  EARLY STOPPING at epoch {epoch + 1}")
            print(f"  Best accuracy: {best_acc:.4f} at epoch {best_epoch}")
            break

    total_time = time.time() - start_time

    model.load_state_dict(best_model_wts)

    final_path = os.path.join(save_dir, f'{model_name}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'history': history,
        'training_time': total_time
    }, final_path)

    print(f"\n  Training complete for {model_name.upper()}")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Best epoch: {best_epoch}, Best val acc: {best_acc:.4f}")

    return model, history, best_acc, total_time


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
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
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

def plot_training_history(history, model_name, save_dir):
    """Plot and save training history."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name.upper()} - Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
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

    # Create model
    model = get_model(model_name, Config.NUM_CLASSES, pretrained=True)
    total_params, trainable_params = count_parameters(model)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Train
    model, history, best_val_acc, training_time = train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=Config.NUM_EPOCHS,
        device=device,
        save_dir=Config.SAVE_DIR
    )

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device)

    # Print final results
    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS: {model_name.upper()}")
    print("=" * 70)
    print(f"  Validation Accuracy:  {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"  Test Accuracy:        {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)")
    print(f"  Top-5 Accuracy:       {test_metrics['top5_accuracy']:.4f} ({test_metrics['top5_accuracy'] * 100:.2f}%)")
    print(f"  Precision:            {test_metrics['precision']:.4f}")
    print(f"  Recall:               {test_metrics['recall']:.4f}")
    print(f"  F1 Score:             {test_metrics['f1_score']:.4f}")
    print(f"  Training Time:        {training_time / 60:.1f} minutes")

    # Plot
    plot_training_history(history, model_name, Config.SAVE_DIR)
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
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'batch_size': Config.BATCH_SIZE,
        'epochs_trained': len(history['train_loss']),
        'val_accuracy': best_val_acc,
        'test_accuracy': test_metrics['accuracy'],
        'top5_accuracy': test_metrics['top5_accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1_score': test_metrics['f1_score'],
        'training_time_min': training_time / 60,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    results_csv_path = os.path.join(Config.SAVE_DIR, 'model_comparison.csv')
    if os.path.exists(results_csv_path):
        existing = pd.read_csv(results_csv_path)
        existing = existing[existing['model'] != model_name]
        new_df = pd.concat([existing, pd.DataFrame([results])], ignore_index=True)
    else:
        new_df = pd.DataFrame([results])

    new_df.to_csv(results_csv_path, index=False)
    print(f"\n  Results saved to: {results_csv_path}")

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