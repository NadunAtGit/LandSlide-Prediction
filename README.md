# Vision-Based Landslide Forecasting Using Satellite Imagery

Predicts landslide susceptibility from **pre-event** Sentinel-2 imagery combined with DEM-derived terrain layers. Case study: Cyclone Ditwah, Sri Lanka, November 2025.

> **Experimental academic system. NOT a certified or operational disaster-warning tool.**

**Key result:** 0.828 ROC-AUC on a geographically held-out test set of 637 patches. 221 of 286 landslides caught at the chosen operating point.

---

## 1. Prerequisites

- **Python 3.11 or newer** (tested on 3.13)
- Git
- VS Code with the **Python** and **Jupyter** extensions
- Internet access to `planetarycomputer.microsoft.com` for satellite and DEM downloads
- Optional: a Google account for Colab GPU training

Check your version:

```
python --version        # Windows
python3 --version       # macOS / Linux
```

---

## 2. Environment setup

```
git clone <REPO_URL>
cd LandSlide-Prediction
```

**Windows (PowerShell)**

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 3. Additional libraries required

The base `requirements.txt` does **not** include everything the notebooks need. Install these as well:

```
pip install scipy scikit-learn scikit-image torch torchvision
```

| Package | Used in | Purpose |
|---|---|---|
| `scipy` | 04, 06 | `cKDTree` for nearest-neighbour thinning; `mannwhitneyu` significance test |
| `scikit-learn` | 08, 09, 10 | Logistic regression, Random Forest, SVM, all evaluation metrics |
| `scikit-image` | 08 | GLCM texture features (`graycomatrix`, `graycoprops`) |
| `torch` | 09, 10 | CNN and MLP training |
| `torchvision` | 09 | ResNet-18 transfer learning |

**Append to `requirements.txt`:**

```
# --- added: machine learning and analysis ---
scipy
scikit-learn
scikit-image
torch
torchvision
```

> `torch` and `torchvision` must be a **matched pair**. Let pip resolve them together. Pinning one and not the other installs cleanly and then throws cryptic C++ symbol errors at import time.

### Optional extras

Not required for the current pipeline, listed for completeness:

```
# alternative DEM download path (the notebooks use rasterio.merge instead)
pip install odc-stac odc-geo

# prototype interface — designed but not yet built
pip install streamlit folium streamlit-folium
```

### GPU (optional)

CPU training works but is slow. For an NVIDIA GPU:

```
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Otherwise run notebook 09 on Google Colab — see section 7.

---

## 4. Jupyter kernel

Register the virtual environment so notebooks use the right interpreter:

```
python -m ipykernel install --user --name landslide --display-name "Python (landslide)"
```

In each notebook: kernel picker (top right) → **Python (landslide)**.

Every notebook's first cell asserts that the kernel path contains `.venv`. If that assertion fails, the wrong interpreter is selected and every import will fail.

Recommended, to keep notebook diffs readable in Git:

```
pip install nbstripout
nbstripout --install
```

---

## 5. Verify the installation

```
python -c "import geopandas, rasterio, torch, sklearn, skimage, scipy; print('all core imports OK')"
```

Then confirm satellite access works before running anything long:

```
python -c "import pystac_client, planetary_computer as pc; c = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace); print(len(list(c.search(collections=['sentinel-2-l2a'], bbox=[80.5,7.0,80.8,7.3], datetime='2025-10-01/2025-11-20').items())), 'scenes found')"
```

A scene count means the acquisition pipeline is viable. A hang or an error means a network or proxy problem to resolve first.

---

## 6. Data setup

Place the landslide inventory supplied by the lecturer here:

```
data/raw/kml/<inventory>.kml      or      .kmz
```

Only one file should be in that folder — the parser takes the first match.

Everything else is downloaded by the notebooks. Nothing else needs to be placed manually.

---

## 7. Running the pipeline

Run the notebooks in order. Notebooks 01 to 08 and 10 run locally; notebook 09 is best run on Colab.

| # | Notebook | Does | Output | Time |
|---|---|---|---|---|
| 01 | `01_inventory` | Parse KML, clean polygons | `landslide_coordinates.csv` | 2 min |
| 04 | `04_negative_coordinates` | Thin positives, sample negatives | `all_coordinates.csv` | 5 min |
| 05 | `05_download_pre_event` | Download Sentinel-2 patches | 3,020 GeoTIFFs | 2–4 hr |
| 06 | `06_add_terrain` | DEM mosaic, slope/hillshade/curvature | 3,024 GeoTIFFs | 30 min |
| 07 | `07_build_dataset` | Stack channels, spatial split, normalise | `X.npy`, `y.npy`, `split.csv` | 10 min |
| 08 | `08_baselines` | Majority, slope-only LR, RF, SVM | `baselines.csv` | 15 min |
| 09 | `09_train_cnn` | CNN, ablations, MLP, ResNet-18 | `cnn_full.pt` | 25 min (GPU) |
| 10 | `10_evaluate` | Threshold, confusion matrix, Grad-CAM | figures, `test_errors.csv` | 5 min |

Notebook 03 (`03_download_post_event`) is retained for label verification only. Its output does not train the forecasting model.

**Notebook 05 is resumable.** It appends to a progress CSV and skips completed locations, so it is safe to interrupt and re-run.

### Running notebook 09 on Colab

1. Upload `X.npy`, `y.npy`, `dataset_index.csv`, `norm_stats.json` and `baselines.csv` to a Google Drive folder named `landslide`
2. Open notebook 09 in Colab, then **Runtime → Change runtime type → T4 GPU**
3. Run all cells; results are written back to Drive under `landslide/outputs/`
4. Download `outputs/` into the local repo before running notebook 10

---

## 8. Project structure

```
data/
  raw/kml/                 landslide inventory (place file here)
  coordinates/             landslide, negative and combined coordinate CSVs
  pre_event/native_tif/    Sentinel-2 patches, 4 bands, 64x64
  pre_event/png_256/       quick-look previews
  terrain/dem_mosaic.tif   cached regional DEM
  terrain/patches/         slope, hillshade, curvature per patch
  terrain/contours/        rendered contour maps (UI and ablation only)
  metadata/                acquisition and terrain feature logs
  processed/               X.npy, y.npy, dataset_index.csv, norm_stats.json
  splits/split.csv         FROZEN spatial split
notebooks/                 01 to 10
outputs/
  models/                  trained weights
  metrics/                 result tables
  figures/                 all generated figures
```

---

## 9. Troubleshooting

**`ModuleNotFoundError: No module named 'fiona'`**
Modern GeoPandas uses `pyogrio` by default. Either `pip install fiona`, or read the KML through pyogrio.

**`DataSourceError: ... is not a valid kmz file`**
On Windows the `zip://` prefix produces backslashes that GDAL's virtual filesystem rejects. Open the `.kmz` path directly, or use `/vsizip/{path.as_posix()}`.

**`AttributeError: 'numpy.ndarray' object has no attribute 'ptp'`**
NumPy 2.x removed the array method. Use `np.ptp(x)` instead of `x.ptp()`.

**`AttributeError: 'numpy.ndarray' object has no attribute 'median'`**
Same cause. Use `np.median(x)`.

**Kernel dies while building the DEM mosaic**
The mosaic is too large for memory. Keep `DEM_RES_M = 30`, the native GLO-30 resolution. A 10 m mosaic needs several gigabytes and adds no real detail.

**Notebook uses the wrong Python**
The first cell prints `sys.executable`. If it does not contain `.venv`, re-select the kernel and restart.

**Downloads are slow**
Notebook 06 caches one regional DEM rather than fetching per patch. Do not remove that cache step.

---

## 10. Working rules

- Branch per person: `git checkout -b <name>/<task>`
- Commit every 60–90 minutes with descriptive messages
  - Good: `Add slope-matched negative sampling with 500m buffer`
  - Bad: `update`, `changes`, `final`
- Commit from your own account — contributions are assessed individually
- Never commit `.tif`, `.npy`, model weights, `.env` or API keys
- **`data/splits/split.csv` is frozen.** Regenerating it after training has begun invalidates every result produced so far
- Do not evaluate on the test split until the evaluation stage

---

## 11. Acknowledgements

**Landslide inventory:** Landslide boundary demarcation dataset prepared by the **Arthur C. Clarke Institute for Modern Technologies (ACCIMT)**, the nationally mandated institution for space science activities in Sri Lanka. Prepared by Mahesh Chathurange and W.G.N.N. Jayawardhana, Space Applications Division, mapped from Sentinel-2 imagery using remote sensing and GIS techniques. Academic, research, planning and decision-support use only. The boundaries are not legally or operationally verified information.

**Imagery:** Contains modified Copernicus Sentinel-2 data (2025) and Copernicus DEM GLO-30, accessed through the Microsoft Planetary Computer STAC API.

---

## 12. Declaration of AI tools and external resources

Generative AI tools were used during this project in the following ways:

- Generating and formatting documentation, including this README and the project report
- Producing presentation slides from results the team had already obtained
- Debugging assistance for environment and library errors
- Explaining remote-sensing concepts and reviewing code for errors

All technical decisions were made by the team, including the forecasting framing and the pre-event cutoff, the negative sampling strategy, the spatial thinning and block-splitting approach, channel selection, model architecture and hyperparameters, and the interpretation of all results. All generated code was reviewed, executed and verified by the team, and every reported metric was produced by running the pipeline in this repository.

**External resources:** Microsoft Planetary Computer STAC API; Copernicus Sentinel-2 and Copernicus DEM GLO-30; Google Colab (T4 GPU) for model training.

---

## 13. Team

| Name | Responsibility |
|---|---|
|  | Data acquisition |
|  | Inventory parsing and sampling |
|  | Terrain derivation |
|  | Model training |
|  | Evaluation and reporting |