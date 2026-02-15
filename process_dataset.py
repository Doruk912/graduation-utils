# ============================================================
# ROBUST RESUME-CAPABLE PREPROCESSING (FAST RESUME)
# ============================================================
# - Fast resume: checks file existence + size only (no image read)
# - Full verification only during processing
# - Progress bar during scanning
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import gc
import uuid

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = '/content/drive/MyDrive/Bird Dataset'
OUTPUT_FOLDER = '/content/drive/MyDrive/Bird Dataset Processed'

TARGET_SIZE = 512
JPEG_QUALITY = 95

# Minimum file size to consider "valid" during quick scan (in bytes)
# 512x512 JPEG at quality 95 should be at least 10KB
MIN_FILE_SIZE = 10000  # 10KB

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def quick_check_exists(output_path, min_size=MIN_FILE_SIZE):
    """
    FAST check if output exists and has reasonable size.
    Does NOT read the image - just checks file metadata.
    Used for resume scanning.
    """
    try:
        if not os.path.exists(output_path):
            return False

        file_size = os.path.getsize(output_path)
        return file_size >= min_size
    except:
        return False


def verify_image_full(filepath, expected_size=None):
    """
    THOROUGH verification - reads and validates image content.
    Used only during processing, not during resume scan.
    """
    try:
        if not os.path.exists(filepath):
            return False, "File does not exist"

        file_size = os.path.getsize(filepath)
        if file_size < 1000:
            return False, f"File too small ({file_size} bytes)"

        img = cv2.imread(filepath)
        if img is None:
            return False, "OpenCV cannot read file"

        h, w = img.shape[:2]
        if expected_size and (h != expected_size or w != expected_size):
            return False, f"Wrong dimensions: {w}x{h}"

        if np.std(img) < 1:
            return False, "Image appears blank"

        if len(img.shape) != 3 or img.shape[2] != 3:
            return False, f"Invalid channels: {img.shape}"

        return True, "OK"

    except Exception as e:
        return False, f"Verification error: {str(e)}"


def process_and_save_image(input_path, output_path, target_size, quality):
    """
    Process a single image with corruption prevention.
    """
    temp_path = None

    try:
        # Read input
        img = cv2.imread(input_path)
        if img is None:
            return False, "Cannot read input"

        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return False, "Input too small"

        if np.std(img) < 1:
            return False, "Input is blank"

        # Resize
        if h > target_size or w > target_size:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC

        img_resized = cv2.resize(img, (target_size, target_size), interpolation=interp)
        del img

        # Write to temp file
        output_dir = os.path.dirname(output_path)
        temp_filename = f".temp_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(output_dir, temp_filename)

        write_success = cv2.imwrite(
            temp_path,
            img_resized,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        del img_resized

        if not write_success:
            return False, "cv2.imwrite failed"

        # Verify temp file
        is_valid, error_msg = verify_image_full(temp_path, expected_size=target_size)
        if not is_valid:
            try:
                os.remove(temp_path)
            except:
                pass
            return False, f"Verification failed: {error_msg}"

        # Atomic rename
        if os.path.exists(output_path):
            os.remove(output_path)

        os.rename(temp_path, output_path)

        # Final check
        if not os.path.exists(output_path):
            return False, "Rename failed"

        return True, None

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False, str(e)

# ============================================================
# MAIN SCRIPT
# ============================================================

print("=" * 60)
print("ROBUST BIRD DATASET PREPROCESSING (FAST RESUME)")
print("=" * 60)
print("• Fast resume scan (no image reading)")
print("• Full verification during processing only")
print("• Safe for runtime disconnections")
print("=" * 60)

total_start = time.time()

# Verify input folder
if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get species folders
species_folders = sorted([f for f in os.listdir(INPUT_FOLDER)
                          if os.path.isdir(os.path.join(INPUT_FOLDER, f))])

print(f"\nConfiguration:")
print(f"  Input:        {INPUT_FOLDER}")
print(f"  Output:       {OUTPUT_FOLDER}")
print(f"  Target size:  {TARGET_SIZE}×{TARGET_SIZE}")
print(f"  JPEG quality: {JPEG_QUALITY}")
print(f"  Species:      {len(species_folders)}")

# ============================================================
# FAST SCAN - IDENTIFY WORK TO DO
# ============================================================

print("\n" + "-" * 60)
print("Fast scanning (checking file existence only)...")
print("-" * 60)

scan_start = time.time()

all_tasks = []
already_done = 0
species_stats = {}

# Count total images first for progress bar
print("Counting images...")
total_to_scan = 0
for species in species_folders:
    input_species_folder = os.path.join(INPUT_FOLDER, species)
    input_images = [f for f in os.listdir(input_species_folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    total_to_scan += len(input_images)

print(f"Total images to check: {total_to_scan:,}")

# Now scan with progress bar
with tqdm(total=total_to_scan, desc="Scanning", unit="img") as pbar:
    for species in species_folders:
        input_species_folder = os.path.join(INPUT_FOLDER, species)
        output_species_folder = os.path.join(OUTPUT_FOLDER, species)

        # Create output folder
        os.makedirs(output_species_folder, exist_ok=True)

        # Get input images
        input_images = [f for f in os.listdir(input_species_folder)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

        species_total = len(input_images)
        species_done = 0
        species_todo = 0

        for img_name in input_images:
            input_path = os.path.join(input_species_folder, img_name)
            output_name = os.path.splitext(img_name)[0] + '.jpg'
            output_path = os.path.join(output_species_folder, output_name)

            # FAST check - only file existence and size (no image read!)
            if quick_check_exists(output_path):
                already_done += 1
                species_done += 1
            else:
                all_tasks.append({
                    'input': input_path,
                    'output': output_path,
                    'species': species,
                    'name': img_name
                })
                species_todo += 1

            pbar.update(1)

        species_stats[species] = {
            'total': species_total,
            'done': species_done,
            'todo': species_todo
        }

scan_time = time.time() - scan_start

# Print species summary
print(f"\nScan completed in {scan_time:.1f} seconds")
print("\nPer-species status:")
print("-" * 50)

for species in species_folders:
    stats = species_stats[species]
    if stats['todo'] == 0:
        print(f"  ✓ {species}: COMPLETE ({stats['done']}/{stats['total']})")
    else:
        pct = stats['done'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  ○ {species}: {stats['done']}/{stats['total']} ({pct:.0f}%), {stats['todo']} remaining")

print("-" * 50)
print(f"  Already processed: {already_done:,}")
print(f"  To process now:    {len(all_tasks):,}")

# ============================================================
# PROCESS REMAINING IMAGES
# ============================================================

if len(all_tasks) == 0:
    print("\n" + "=" * 60)
    print("✓ ALL IMAGES ALREADY PROCESSED!")
    print("=" * 60)

else:
    print("\n" + "-" * 60)
    print(f"Processing {len(all_tasks):,} images...")
    print("-" * 60)
    print()

    success_count = 0
    fail_count = 0
    failed_files = []

    process_start = time.time()
    last_gc = 0

    with tqdm(total=len(all_tasks), desc="Processing", unit="img",
              dynamic_ncols=True, smoothing=0.02) as pbar:

        for i, task in enumerate(all_tasks):
            success, error = process_and_save_image(
                task['input'],
                task['output'],
                TARGET_SIZE,
                JPEG_QUALITY
            )

            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append({
                    'input': task['input'],
                    'name': task['name'],
                    'species': task['species'],
                    'error': error
                })

            pbar.update(1)

            # Update stats periodically
            if (i + 1) % 50 == 0:
                elapsed = time.time() - process_start
                rate = (success_count + fail_count) / elapsed if elapsed > 0 else 0
                remaining = len(all_tasks) - (i + 1)
                eta_min = remaining / rate / 60 if rate > 0 else 0
                pbar.set_postfix({
                    'ok': success_count,
                    'fail': fail_count,
                    'rate': f'{rate:.1f}/s',
                    'ETA': f'{eta_min:.0f}m'
                })

            # Garbage collection
            if (i + 1) - last_gc >= 200:
                gc.collect()
                last_gc = i + 1

    process_time = time.time() - process_start

    # ============================================================
    # RETRY FAILED IMAGES
    # ============================================================

    if failed_files:
        print(f"\n⚠ {len(failed_files)} images failed. Retrying...")

        retry_success = 0
        still_failed = []

        for item in tqdm(failed_files, desc="Retrying", unit="img"):
            time.sleep(0.1)  # Small delay helps with Drive I/O

            success, error = process_and_save_image(
                item['input'],
                item['output'] if 'output' in item else os.path.join(
                    OUTPUT_FOLDER,
                    item['species'],
                    os.path.splitext(item['name'])[0] + '.jpg'
                ),
                TARGET_SIZE,
                JPEG_QUALITY
            )

            if success:
                retry_success += 1
            else:
                still_failed.append({
                    'path': item['input'],
                    'error': error
                })

        print(f"  Retry recovered: {retry_success} images")
        success_count += retry_success
        fail_count = len(still_failed)
        failed_files = still_failed

    # ============================================================
    # SESSION SUMMARY
    # ============================================================

    print("\n" + "-" * 60)
    print("SESSION COMPLETE")
    print("-" * 60)

    avg_rate = (success_count + fail_count) / process_time if process_time > 0 else 0

    print(f"""
This session:
  • Processed: {success_count:,} images
  • Failed:    {fail_count} images
  • Time:      {process_time/60:.1f} minutes
  • Speed:     {avg_rate:.2f} images/sec
""")

# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET STATUS")
print("=" * 60)

# Quick count of output files
print("\nCounting output files...")
final_counts = {}
total_output = 0

for species in tqdm(species_folders, desc="Counting"):
    output_species = os.path.join(OUTPUT_FOLDER, species)
    if os.path.exists(output_species):
        count = len([f for f in os.listdir(output_species) if f.endswith('.jpg')])
        final_counts[species] = count
        total_output += count
    else:
        final_counts[species] = 0

# Calculate totals
total_expected = sum(s['total'] for s in species_stats.values())
completion_pct = (total_output / total_expected * 100) if total_expected > 0 else 0

total_time = time.time() - total_start

print(f"""
{'=' * 50}
FINAL REPORT
{'=' * 50}

Dataset:
  • Output:    {OUTPUT_FOLDER}
  • Format:    {TARGET_SIZE}×{TARGET_SIZE} JPEG (quality {JPEG_QUALITY})

Progress:
  • Expected:  {total_expected:,} images
  • Completed: {total_output:,} images
  • Progress:  {completion_pct:.1f}%

Total time: {total_time/60:.1f} minutes
""")

# Per-species breakdown
print("Per-species status:")
print("-" * 50)

complete_species = 0
for species in sorted(final_counts.keys()):
    count = final_counts[species]
    expected = species_stats.get(species, {}).get('total', 5000)
    pct = (count / expected * 100) if expected > 0 else 0

    if pct >= 99.5:
        status = "✓"
        complete_species += 1
    elif pct >= 90:
        status = "◐"
    else:
        status = "○"

    print(f"  {status} {species}: {count:,}/{expected:,} ({pct:.0f}%)")

print("-" * 50)
print(f"  Complete: {complete_species}/{len(species_folders)} species")

# Show failed files
if 'failed_files' in dir() and failed_files:
    print(f"\n⚠ Permanently failed ({len(failed_files)}):")
    for item in failed_files[:10]:
        print(f"  • {os.path.basename(item['path'])}: {item['error']}")
    if len(failed_files) > 10:
        print(f"  ... and {len(failed_files) - 10} more")

# ============================================================
# NEXT STEPS
# ============================================================

print("\n" + "=" * 60)

if completion_pct >= 99.5:
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nDataset ready at: {OUTPUT_FOLDER}")

    # Optional: Verify random samples
    print("\nVerifying random samples...")
    import random
    all_outputs = []
    for species in species_folders:
        sp_folder = os.path.join(OUTPUT_FOLDER, species)
        if os.path.exists(sp_folder):
            for f in os.listdir(sp_folder):
                if f.endswith('.jpg'):
                    all_outputs.append(os.path.join(sp_folder, f))

    samples = random.sample(all_outputs, min(100, len(all_outputs)))
    valid_count = 0
    for s in tqdm(samples, desc="Verifying samples"):
        is_valid, _ = verify_image_full(s, TARGET_SIZE)
        if is_valid:
            valid_count += 1

    print(f"  Sample verification: {valid_count}/{len(samples)} OK ({valid_count/len(samples)*100:.0f}%)")

else:
    print("○ PREPROCESSING IN PROGRESS")
    print("=" * 60)
    remaining = total_expected - total_output
    print(f"\nRemaining: {remaining:,} images ({100 - completion_pct:.1f}%)")
    print("\n→ Run this script again to continue!")

print("\n" + "=" * 60)