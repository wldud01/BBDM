# BBDM Info

* **Paper:** BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models (CVPR)

  * [CVPR 2023 Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf)
  * [Official GitHub Repository](https://github.com/xuekt98/BBDM/tree/main)

* **Task:** Image translation (Source → Target)

* **Base Model:** Diffusion Models

## Main Characteristics

* Learns transition path between distributions using Brownian Bridge diffusion process

  * Forward Process: Noise added to Target image to reach Source
  * Reverse Process: Learned to generate Target image from noisy Source
* Works with **paired data**
* Applicable in both **pixel-space** and **latent-space**

  * If using latent space, VQGAN's latent representation is adopted

## Implement Detail

* **Current Implementation:** Pixel-space version only

  * Author's code includes both pixel-space and latent-space version

## 📁 Project Structure Overview

### `main.py`

* Entry point: reads `config.yaml`, sets up environment (CPU/GPU/DDP), and runs training/inference using specified `Runner`

### `prepare_data_folder.py`

* Generates training folders from provided preprocessed images
* Folders created:

  * `train/`, `val/`, `test/` (ratio: 0.8 / 0.1 / 0.1)

### `Register.py`

* Dynamic registration utility to manage different model runners via config

### `utils.py`

* Converts `DictConfig` ⇒ `Namespace` and vice versa
* Dynamically creates Python objects (e.g. Runner class) from string path
* Returns instantiated `Runner` from config

## 📁 `configs`

* `Template-BBDM.yaml`: Configuration for pixel-space BBDM
* Other YAML files: Latent-space settings

## 📁 `datasets`

### `custom.py`

* Loads NECT → CECT image pairs for training

* **Required folder structure:**

  ```
  dataset_path/
    train/
      NECT/
        S0001_NECT_0001.png
        ...
      CECT/
        S0001_CECT_0001.png
        ...
    val/
      NECT/
      CECT/
    test/
      NECT/
      CECT/
  ```

* Dataset creation utility: `prepare_data_folder.py`

### `base.py`

* Resize and flip-based image augmentation (doubles dataset if enabled)
* Normalize images to \[-1, 1] range

### `utils.py`

* Recursively scans all images in folders and returns path lists

## 📁 `model`

### `utils.py`

* `extract()`, `exists()`, `default()` used for indexing, checking, or defaulting values (e.g. noise schedule)

### `BrownianBridgeModel.py`

* Core diffusion process logic (forward, reverse, sampling, training)

### `base/utils.py`

* Logging, image type checks, defaulting, parameter counting, dynamic instantiation

### `base/modules`

* Attention, transformer modules

### `base/modules/diffusionmodules`

* UNet-based architecture components (e.g. `openaimodel.py`)

## 📁 `runners`

### `DiffusionBasedModel`

* `BBDMRunner.py`: Main training/testing logic for BBDM
* `DiffusionBaseRunner.py`: Shared base class for diffusion-based runners

### `base`

* `EMA.py`: Exponential Moving Average for parameter smoothing
* `BaseRunner.py`, `utils.py`: Generic training helpers and base classes
