#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# =========================Pre-Processing of data ===================

# ================ BR Layers Threshold Calc ===============

import numpy as np
import rasterio
from skimage.filters import threshold_otsu

# List of ratio layer file paths
ratio_files = [
    r"E:/Sentinel_Dataset/band_ratios/hydroxyl_B11_div_B12.tif",
    r"E:/Sentinel_Dataset/band_ratios/ferric_B4_div_B3.tif",
    r"E:/Sentinel_Dataset/band_ratios/ferrous_combined.tif",
    r"E:/Sentinel_Dataset/band_ratios/ferricoxide_B11_div_B08.tif"
]

for path in ratio_files:
    with rasterio.open(path) as src:
        arr = src.read(1)  # Read band 1
        nodata = src.nodata
        if nodata is not None:
            arr = arr[arr != nodata]
        arr = arr.astype(float)

    # Compute Otsu threshold
    threshold_value = threshold_otsu(arr)
    print(f"{path} -> Otsu Threshold: {threshold_value:.4f}")





# hydroxyl_B11_div_B12.tif -> Otsu Threshold: 1.0563
# ferric_B4_div_B3.tif -> Otsu Threshold: 1.1499
# ferrous_combined.tif -> Otsu Threshold: 2.0103
# ferricoxide_B11_div_B08.tif -> Otsu Threshold: 1.1777


# In[2]:


# ====================== GOSSAN MAP =========================

import rasterio
import numpy as np

# ===== USER INPUTS =====
hydroxyl_path = r"E:/Sentinel_Dataset/band_ratios/filtered/hydroxyl_filtered.tif"
ferric_path = r"E:/Sentinel_Dataset/band_ratios/filtered/ferric_filtered.tif"
ferrous_path = r"E:/Sentinel_Dataset/band_ratios/filtered/ferrous_filtered.tif"
ferricoxide_path = r"E:/Sentinel_Dataset/band_ratios/filtered/ferricoxide_filtered.tif"

gdi_path = r"E:/Sentinel_Dataset/gossan_detection/gossan_composite_raster_float.tif"
gossan_output_path = r"E:/Sentinel_Dataset/gossan_detection/gossan_composite_raster_binary.tif"

# Thresholds
t_low = 0.94
t_high = 1.06

# ===== STEP 1: READ LAYERS =====
with rasterio.open(hydroxyl_path) as src:
    hydroxyl = src.read(1).astype(np.float32)
    meta = src.meta.copy()
    nodata_val = src.nodata
    if nodata_val is not None:
        hydroxyl = np.where(hydroxyl == nodata_val, 0, hydroxyl)

with rasterio.open(ferric_path) as src:
    ferric = src.read(1).astype(np.float32)
    if src.nodata is not None:
        ferric = np.where(ferric == src.nodata, 0, ferric)

with rasterio.open(ferrous_path) as src:
    ferrous = src.read(1).astype(np.float32)
    if src.nodata is not None:
        ferrous = np.where(ferrous == src.nodata, 0, ferrous)

with rasterio.open(ferricoxide_path) as src:
    ferricoxide = src.read(1).astype(np.float32)
    if src.nodata is not None:
        ferricoxide = np.where(ferricoxide == src.nodata, 0, ferricoxide)

# ===== STEP 2: CALCULATE GDI COMPOSITE =====
gdi_composite = (
    (0.4 * ferricoxide) +
    (0.3 * ferric) +
    (0.2 * hydroxyl) +
    (0.1 * ferrous)
).astype(np.float32)

# Save GDI composite
meta.update(dtype=rasterio.float32, count=1, nodata=0)
with rasterio.open(gdi_path, 'w', **meta) as dst:
    dst.write(gdi_composite, 1)

print(f"✅ GDI composite saved: {gdi_path}")

# ===== STEP 3: CLASSIFICATION =====
classified = np.zeros_like(gdi_composite, dtype=np.uint8)
classified[(gdi_composite >= t_low) & (gdi_composite < t_high)] = 1
classified[gdi_composite >= t_high] = 2

# ===== STEP 4: SAVE FINAL GOSSAN MAP =====
meta.update(dtype=rasterio.uint8, count=1, nodata=0)
with rasterio.open(gossan_output_path, 'w', **meta) as dst:
    dst.write(classified, 1)

# ===== STEP 5: PRINT CLASS DISTRIBUTION =====
unique, counts = np.unique(classified, return_counts=True)
print("Pixel counts per class:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c} pixels")

print(f"✅ Gossan map saved: {gossan_output_path}")



# In[26]:


# ================= Normalise Band Ratios =======================

import rasterio
import numpy as np
import os

# Input folder with your ratio layers
input_folder = r"E:\Sentinel_Dataset\band_ratios\filtered"
output_folder = r"E:\Sentinel_Dataset\band_ratios\standardized"
os.makedirs(output_folder, exist_ok=True)

# Loop through all tif files in the folder
for file in os.listdir(input_folder):
    if file.lower().endswith(".tif"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, f"{file}_norm")

        with rasterio.open(input_path) as src:
            data = src.read(1).astype(float)  # Read first band
            profile = src.profile

            # Mask out NoData values
            nodata = profile.get('nodata', None)
            if nodata is not None:
                mask = data == nodata
            else:
                mask = np.isnan(data)

            # Standardization (mean=0, std=1) ignoring NoData
            valid_data = data[~mask]
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)

            standardized_data = (data - mean_val) / std_val

            # Restore NoData
            standardized_data[mask] = nodata if nodata is not None else np.nan

            # Save output
            profile.update(dtype=rasterio.float32)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(standardized_data.astype(np.float32), 1)

        print(f"Standardized: {file}")


# In[32]:


# ==================== PCA =======================

import rasterio
import numpy as np
from sklearn.decomposition import PCA
import os

# Path to your stacked 4-band raster
stacked_file = r"E:/Sentinel_Dataset/stacked_ratios.tif"

# Output folder
output_dir = r"E:/Sentinel_Dataset"
os.makedirs(output_dir, exist_ok=True)

# Read stacked raster
with rasterio.open(stacked_file) as src:
    profile = src.profile
    stacked = src.read()  # Shape: (bands, rows, cols)

# Rearrange shape to (rows, cols, bands)
stacked = np.transpose(stacked, (1, 2, 0))
rows, cols, bands_count = stacked.shape

# Reshape for PCA: (pixels, bands)
X = stacked.reshape(-1, bands_count)

# Run PCA
pca = PCA(n_components=bands_count)
pca_result = pca.fit_transform(X)

# Reshape each component back to raster form
pca_images = pca_result.reshape(rows, cols, bands_count)

# Save each component as separate GeoTIFF
for i in range(bands_count):
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1)
    out_path = os.path.join(output_dir, f"pca_pc{i+1}.tif")
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(pca_images[:, :, i].astype(np.float32), 1)

print(f"PCA complete. Components saved to {output_dir}")


# In[2]:


# ================= Multiclass Labels + Multiple Inputs Tiling ==================

import rasterio
import numpy as np
import os
from rasterio.windows import Window

# ==== USER INPUTS ====
label_raster = r"E:/Sentinel_Dataset/gossan_detection/gossan_composite_raster_binary.tif"  # binary labels

# Input rasters (total 8)
rasters = {
    "pca": r"E:/Sentinel_Dataset/datasets/pca_1_2_stack_aligned.tif",
    "mnf": r"E:/Sentinel_Dataset/datasets/mnf_1_2_stack_aligned.tif",
    "core": r"E:/Sentinel_Dataset/datasets/core_area_raster_aligned.tif",
    "ratios": r"E:/Sentinel_Dataset/datasets/stacked_ratios_aligned.tif",
    "core_ratios": r"E:/Sentinel_Dataset/datasets/core_and_ratios.tif",
    "core_pca_mnf": r"E:/Sentinel_Dataset/datasets/core_and_pca_and_mnf.tif",
    "core_ratios_pca_mnf": r"E:/Sentinel_Dataset/datasets/core_and_ratios_and_pca_and_mnf.tif",
    "ratios_pca_mnf": r"E:/Sentinel_Dataset/datasets/ratios_and_pca_and_mnf.tif"
}

# Output directories
output_dir_labels = r"E:/Sentinel_Dataset/tiles_64_overlap/labels"
output_dirs_inputs = {name: f"E:/Sentinel_Dataset/tiles_64_overlap/{name}" for name in rasters.keys()}

tile_size = 64 # 64 * 64
strong_threshold = 12  # Minimum % of stronger class pixels for tile to count
weak_threshold = 2     # Minimum % of weaker class pixels for tile to count
overlap = 0.5          # 50% overlap

# ==== CREATE FOLDERS ====
folders_labels = {
    "strong": os.path.join(output_dir_labels, "strong"),
    "weak": os.path.join(output_dir_labels, "weak"),
    "none": os.path.join(output_dir_labels, "none")
}
for f in folders_labels.values():
    os.makedirs(f, exist_ok=True)

for d in output_dirs_inputs.values():
    os.makedirs(d, exist_ok=True)

# ==== READ ALL RASTERS ====
with rasterio.open(label_raster) as src_labels, \
     rasterio.open(rasters["pca"]) as src_pca, \
     rasterio.open(rasters["mnf"]) as src_mnf, \
     rasterio.open(rasters["core"]) as src_core, \
     rasterio.open(rasters["ratios"]) as src_ratios, \
     rasterio.open(rasters["core_ratios"]) as src_core_ratios, \
     rasterio.open(rasters["core_pca_mnf"]) as src_core_pca_mnf, \
     rasterio.open(rasters["core_ratios_pca_mnf"]) as src_core_ratios_pca_mnf, \
     rasterio.open(rasters["ratios_pca_mnf"]) as src_ratios_pca_mnf:

    width = src_labels.width
    height = src_labels.height
    profile_labels = src_labels.profile

    # Store rasterio sources in dict for easier looping
    sources = {
        "pca": src_pca,
        "mnf": src_mnf,
        "core": src_core,
        "ratios": src_ratios,
        "core_ratios": src_core_ratios,
        "core_pca_mnf": src_core_pca_mnf,
        "core_ratios_pca_mnf": src_core_ratios_pca_mnf,
        "ratios_pca_mnf": src_ratios_pca_mnf
    }
    profiles = {name: src.profile.copy() for name, src in sources.items()}

    step_size = int(tile_size * (1 - overlap))
    tile_id = 0

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)

            if w < tile_size or h < tile_size:
                continue  # Skip incomplete tiles at edges

            window = Window(x, y, w, h)

            # === READ LABEL TILE ===
            label_tile = src_labels.read(1, window=window)

            total_pixels = label_tile.size
            strong_count = np.count_nonzero(label_tile == 2)
            weak_count = np.count_nonzero(label_tile == 1)

            strong_percent = (strong_count / total_pixels) * 100
            weak_percent = (weak_count / total_pixels) * 100

            # === CLASSIFY TILE ===
            if strong_percent >= strong_threshold:
                folder = folders_labels["strong"]
            elif weak_percent >= weak_threshold:
                folder = folders_labels["weak"]
            else:
                folder = folders_labels["none"]

            # === SAVE LABEL TILE ===
            tile_id += 1
            profile_labels.update({
                'width': w,
                'height': h,
                'count': 1,
                'transform': rasterio.windows.transform(window, src_labels.transform)
            })
            out_label = os.path.join(folder, f"tile_{tile_id}.tif")
            with rasterio.open(out_label, 'w', **profile_labels) as dst:
                dst.write(label_tile, 1)

            # === SAVE ALL INPUT RASTER TILES ===
            for name, src in sources.items():
                tile = src.read(window=window)  # keep all bands
                profiles[name].update({
                    'width': w,
                    'height': h,
                    'transform': rasterio.windows.transform(window, src.transform)
                })
                out_path = os.path.join(output_dirs_inputs[name], f"tile_{tile_id}.tif")
                with rasterio.open(out_path, 'w', **profiles[name]) as dst:
                    dst.write(tile)

# ==== COUNT RESULTING TILES ====
count_strong = len(os.listdir(folders_labels["strong"]))
count_weak = len(os.listdir(folders_labels["weak"]))
count_none = len(os.listdir(folders_labels["none"]))

print("Tiling complete with 8 aligned sample inputs and labels!")
print(f"Total tiles created: {tile_id}")
print(f"Strong class tiles: {count_strong}")
print(f"Weak class tiles: {count_weak}")
print(f"None class tiles: {count_none}")




# ========================= MAKE_SAMPLE_LABEL_PAIRS =========================

import os
import shutil
import pathlib
import random

# ======================= DIRECTORY PATH SETUP =======================

# Input sample path (change "core" -> "ratios"/"pca"/etc next time)
sample_dir = pathlib.Path("E:/Sentinel_Dataset/tiles_64_overlap/samples/ratios_pca_mnf")
label_dir = pathlib.Path("E:/Sentinel_Dataset/tiles_64_overlap/labels")

# Base output directory for paired dataset
new_dir = pathlib.Path("E:/Sentinel_Dataset/cnn/model_datasets/ratios_pca_mnf_dataset")

# ======================= TRAIN/VAL/TEST SPLITS =======================
train_split = 0.70
val_split = 0.15
test_split = 0.15

# ======================= COLLECT FILES PER CATEGORY =======================
category_files = {}
for category in ("none", "weak", "strong"):
    cat_dir = label_dir / category
    files = sorted(os.listdir(cat_dir))
    random.shuffle(files)  # shuffle within class
    category_files[category] = files

# ======================= SPLIT PER CATEGORY =======================
splits = {"train": {}, "validation": {}, "test": {}}

for category, files in category_files.items():
    total_count = len(files)
    train_count = int(total_count * train_split)
    val_count   = int(total_count * val_split)
    test_count  = total_count - train_count - val_count

    splits["train"][category] = files[:train_count]
    splits["validation"][category] = files[train_count:train_count + val_count]
    splits["test"][category] = files[train_count + val_count:]

# ======================= SUBSET CREATION FUNCTION =======================
def make_subset(subset_name, split_dict):
    for category in ("none", "weak", "strong"):
        dir = new_dir / subset_name / category
        os.makedirs(dir, exist_ok=True)

        for fname in split_dict[category]:
            src_sample = sample_dir / fname
            dst_sample = dir / fname
            if src_sample.exists():
                shutil.copyfile(src_sample, dst_sample)
            else:
                print(f"Warning: Sample not found for {fname}")

# ======================= MAKE SUBSETS =======================
make_subset("train", splits["train"])
make_subset("validation", splits["validation"])
make_subset("test", splits["test"])

# ======================= REPORT =======================
print("Balanced dataset preparation complete!")
for subset in ("train", "validation", "test"):
    total = sum(len(splits[subset][cat]) for cat in ("none","weak","strong"))
    print(f"{subset.capitalize()} total: {total} "
          f"(none={len(splits[subset]['none'])}, "
          f"weak={len(splits[subset]['weak'])}, "
          f"strong={len(splits[subset]['strong'])})")


# In[155]:


from sklearn.metrics import classification_report
import numpy as np

# Get predictions on test set
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, digits=3))


# In[157]:


images, labels = next(iter(val_dataset))
preds = model.predict(images)
print("Sample predictions:", np.argmax(preds, axis=1))
print("True labels:", labels.numpy())


# In[159]:


import numpy as np
from collections import Counter

def count_classes(dataset):
    counts = Counter()
    for images, labels in dataset.unbatch():  # remove batching
        counts[int(labels.numpy())] += 1
    return counts

train_counts = count_classes(train_dataset)
val_counts   = count_classes(val_dataset)
test_counts  = count_classes(test_dataset)

print("Train class distribution:", train_counts)
print("Val class distribution:", val_counts)
print("Test class distribution:", test_counts)





