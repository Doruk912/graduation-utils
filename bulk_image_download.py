import os
import pandas as pd
import requests
from google.colab import drive
from tqdm.notebook import tqdm
import time

# 1. MOUNT GOOGLE DRIVE
drive.mount('/content/drive')

# ==========================================
# 2. CONFIGURATION
# ==========================================
BASE_PATH = '/content/drive/MyDrive/Bird Dataset'

# Target number of images per species
TARGET_LIMIT = 1000

# List of species
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

def download_images_for_species(species_name):
    species_folder = os.path.join(BASE_PATH, species_name)
    multimedia_file = os.path.join(species_folder, 'multimedia.txt')

    # Check if folder exists
    if not os.path.exists(species_folder):
        print(f"⚠️ Folder not found: {species_name}. Skipping...")
        return

    # Check current file count to see if we are already done
    # Only count .jpg files to avoid counting the txt files
    existing_images = [f for f in os.listdir(species_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
    current_count = len(existing_images)

    if current_count >= TARGET_LIMIT:
        print(f"✅ {species_name}: Already has {current_count} images. Skipping.")
        return

    print(f"📂 Processing: {species_name} (Found {current_count}/{TARGET_LIMIT} images)")

    # Load the multimedia.txt file
    # Use quoting=3 (QUOTE_NONE) to avoid errors with messy text in descriptions
    try:
        df = pd.read_csv(multimedia_file, sep='\t', on_bad_lines='skip', quoting=3)
    except Exception as e:
        print(f"❌ Error reading multimedia.txt for {species_name}: {e}")
        return

    # Filter for images only and ensure valid identifiers
    if 'type' in df.columns and 'identifier' in df.columns:
        # Get only StillImage rows that have a URL
        images_df = df[
            (df['type'] == 'StillImage') &
            (df['identifier'].notna())
        ]
    else:
        print(f"❌ Columns 'type' or 'identifier' missing in {species_name}")
        return

    # Iterate and Download
    # Use tqdm to show a progress bar
    with tqdm(total=TARGET_LIMIT, initial=current_count, desc=species_name) as pbar:

        for index, row in images_df.iterrows():
            # Stop if the limit is reached
            if current_count >= TARGET_LIMIT:
                break

            img_url = row['identifier']
            gbif_id = str(row['gbifID'])

            # Create a unique filename using the gbifID
            file_name = f"{gbif_id}.jpg"
            file_path = os.path.join(species_folder, file_name)

            # CHECK: Does file already exist?
            if os.path.exists(file_path):
                # If it exists, verify it's not empty (0 bytes)
                if os.path.getsize(file_path) > 0:
                    continue # It's good, skip to next
                else:
                    # File exists but is empty (corrupt), re-download
                    pass

            # DOWNLOAD
            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    current_count += 1
                    pbar.update(1)
                else:
                    # URL might be dead, just skip
                    pass
            except Exception as e:
                # Connection error, skip image
                pass

            # Be polite to the server
            time.sleep(0.1)

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
print("🚀 Starting Download Process...")
for species in SPECIES_LIST:
    download_images_for_species(species)
print("\n🎉 All downloads finished!")