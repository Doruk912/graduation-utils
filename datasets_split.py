# ============================================================
# TRAIN/VAL/TEST SPLIT CREATION
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
BASE_PATH = r'C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed'
OUTPUT_PATH = r'C:\Users\Doruk\OneDrive\Masaüstü\Graduation Project\Bird Dataset Processed'  # Save CSV in same location

SPECIES_LIST = [
    "Larus michahellis", "Corvus cornix", "Passer domesticus",
    "Spilopelia senegalensis", "Columba livia", "Fringilla coelebs",
    "Anas platyrhynchos", "Ardea cinerea", "Phalacrocorax carbo",
    "Streptopelia decaocto", "Pica pica", "Chroicocephalus ridibundus",
    "Garrulus glandarius", "Sturnus vulgaris", "Motacilla alba",
    "Parus major", "Ciconia ciconia", "Fulica atra",
    "Hirundo rustica", "Coloeus monedula", "Psittacula krameri",
    "Turdus merula", "Acridotheres tristis"
]

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42  # For reproducibility

print("=" * 70)
print("CREATING TRAIN/VAL/TEST SPLIT")
print("=" * 70)
print(f"Split ratios: Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")
print(f"Random seed: {RANDOM_SEED}")
print()

# ============================================================
# STEP 1: Collect all image paths and labels
# ============================================================

print("Step 1: Collecting image paths...")

data = []
for species in SPECIES_LIST:
    species_path = os.path.join(BASE_PATH, species)
    if not os.path.exists(species_path):
        print(f"  ⚠ Warning: {species} folder not found")
        continue

    images = [f for f in os.listdir(species_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    for img_name in images:
        data.append({
            'image_path': os.path.join(species_path, img_name),
            'image_name': img_name,
            'species': species
        })

df = pd.DataFrame(data)
print(f"  Total images found: {len(df):,}")
print(f"  Species: {df['species'].nunique()}")

# ============================================================
# STEP 2: Create label encoding
# ============================================================

print("\nStep 2: Creating label encoding...")

# Sort species alphabetically for consistent encoding
species_sorted = sorted(df['species'].unique())
species_to_idx = {species: idx for idx, species in enumerate(species_sorted)}
idx_to_species = {idx: species for species, idx in species_to_idx.items()}

df['label'] = df['species'].map(species_to_idx)

print(f"  Labels assigned: 0 to {len(species_to_idx) - 1}")

# ============================================================
# STEP 3: Stratified split
# ============================================================

print("\nStep 3: Performing stratified split...")

# First split: separate test set
train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_RATIO,
    stratify=df['species'],
    random_state=RANDOM_SEED
)

# Second split: separate train and validation from remaining
val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_ratio_adjusted,
    stratify=train_val_df['species'],
    random_state=RANDOM_SEED
)

# Assign split labels
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

print(f"  Train: {len(train_df):,} images ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df):,} images ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df):,} images ({len(test_df)/len(df)*100:.1f}%)")

# ============================================================
# STEP 4: Verify stratification
# ============================================================

print("\nStep 4: Verifying stratification (images per species)...")

print(f"\n{'Species':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print("-" * 70)

for species in species_sorted:
    train_count = len(train_df[train_df['species'] == species])
    val_count = len(val_df[val_df['species'] == species])
    test_count = len(test_df[test_df['species'] == species])
    total = train_count + val_count + test_count
    print(f"{species:<30} {train_count:>8} {val_count:>8} {test_count:>8} {total:>8}")

print("-" * 70)
print(f"{'TOTAL':<30} {len(train_df):>8} {len(val_df):>8} {len(test_df):>8} {len(df):>8}")

# ============================================================
# STEP 5: Combine and save
# ============================================================

print("\nStep 5: Saving split information...")

# Combine all splits
final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Save main CSV
csv_path = os.path.join(OUTPUT_PATH, 'dataset_split.csv')
final_df.to_csv(csv_path, index=False)
print(f"  ✓ Saved: {csv_path}")

# Save label mapping
label_map_df = pd.DataFrame([
    {'label': idx, 'species': species}
    for species, idx in species_to_idx.items()
])
label_map_path = os.path.join(OUTPUT_PATH, 'label_mapping.csv')
label_map_df.to_csv(label_map_path, index=False)
print(f"  ✓ Saved: {label_map_path}")

# Save separate CSVs for convenience
train_df.to_csv(os.path.join(OUTPUT_PATH, 'train.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_PATH, 'val.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_PATH, 'test.csv'), index=False)
print(f"  ✓ Saved: train.csv, val.csv, test.csv")

# ============================================================
# STEP 6: Visualize split distribution
# ============================================================

print("\nStep 6: Visualizing split distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Overall split
split_counts = [len(train_df), len(val_df), len(test_df)]
split_labels = ['Train\n(80%)', 'Validation\n(10%)', 'Test\n(10%)']
colors = ['#2ecc71', '#3498db', '#e74c3c']

axes[0].pie(split_counts, labels=split_labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.02, 0.02, 0.02))
axes[0].set_title('Overall Dataset Split', fontsize=12, fontweight='bold')

# Plot 2: Per-species distribution
species_short = [s.split()[-1][:10] for s in species_sorted]  # Shortened names
train_counts = [len(train_df[train_df['species'] == s]) for s in species_sorted]
val_counts = [len(val_df[val_df['species'] == s]) for s in species_sorted]
test_counts = [len(test_df[test_df['species'] == s]) for s in species_sorted]

x = np.arange(len(species_sorted))
width = 0.6

axes[1].bar(x, train_counts, width, label='Train', color='#2ecc71')
axes[1].bar(x, val_counts, width, bottom=train_counts, label='Val', color='#3498db')
axes[1].bar(x, test_counts, width, bottom=np.array(train_counts) + np.array(val_counts),
            label='Test', color='#e74c3c')

axes[1].set_xlabel('Species')
axes[1].set_ylabel('Number of Images')
axes[1].set_title('Split Distribution per Species', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(species_short, rotation=45, ha='right', fontsize=7)
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'split_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE - SUMMARY")
print("=" * 70)
print(f"""
Dataset: {BASE_PATH}
Total Images: {len(df):,}
Classes: {len(species_to_idx)}

Split Distribution:
┌─────────────┬─────────────┬──────────────────┐
│ Split       │ Images      │ Per Class        │
├─────────────┼─────────────┼──────────────────┤
│ Train       │ {len(train_df):>9,}   │ ~{len(train_df)//23:,}            │
│ Validation  │ {len(val_df):>9,}   │ ~{len(val_df)//23:,}             │
│ Test        │ {len(test_df):>9,}   │ ~{len(test_df)//23:,}             │
├─────────────┼─────────────┼──────────────────┤
│ Total       │ {len(df):>9,}   │ 5,000            │
└─────────────┴─────────────┴──────────────────┘

Files Saved:
  • {OUTPUT_PATH}/dataset_split.csv (main file with all splits)
  • {OUTPUT_PATH}/label_mapping.csv (species to label mapping)
  • {OUTPUT_PATH}/train.csv
  • {OUTPUT_PATH}/val.csv
  • {OUTPUT_PATH}/test.csv
  • {OUTPUT_PATH}/split_distribution.png

Random Seed: {RANDOM_SEED} (use same seed to reproduce splits)
""")

print("=" * 70)
print("READY FOR TRAINING!")
print("=" * 70)