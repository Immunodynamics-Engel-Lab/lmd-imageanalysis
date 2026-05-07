# Segmentation and Reference Point Detection for Laser Capture Microdissection (LMD)

## Project Summary

This repository contains the code for a [Cellpose-SAM](https://github.com/MouseLand/cellpose) & [pyLMD](https://github.com/MannLabs/py-lmd) project dedicated to automating **cell boundary and reference point detection** in microscopic images used for **Laser Capture Microdissection (LMD)**.

The primary function of this repository is to identify the boundaries of target cells and detect precise reference points from laser engraved 'T' structures and is designed to enhance the reproducibility of the LMD workflow.

---


## 🔬 Stepwise Workflow

1. **Fiducial Marking**  
   Burn fiducial T-marks into tissue sections using LMD laser.

2. **Image Acquisition**  
   Acquire fluorescence microscopy images.

3. **Segmentation** (Computational part 1)  
   Perform **Cellpose-SAM** segmentation to delineate cell boundaries.

4. **Filtering** (Computational part 1)  
   Apply intensity and overlap filtering to refine segmentation results.

5. **Fiducial Detection** (Computational part 2)  
   Detect fiducials via **FFT-based template matching**.

6. **Data Export** (Computational part 2)  
   Determine and export contours and coordinates via **pyLMD** as XML.

7. **Software Integration**  
   Import XML into **Leica LMD software**.

8. **Alignment & Excision**  
   Align fiducials and perform precise laser excision.

9. **Validation**  
   Validate excision results microscopically.

---

## Getting Started

### 📋 Prerequisites

This project exclusively uses the [**Pixi**](https://pixi.sh/latest/python/tutorial/) package manager to guarantee a reliable and isolated Python environment. [**Install Instructions**](https://prefix-dev.github.io/pixi/main/install.html)

### 💻 Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourUsername/lmd-imageanalysis.git
    cd lmd-imageanalysis
    ```

2.  **Initialize the Pixi Environment:**

    Pixi reads the required dependencies from the `pixi.toml` file and creates a ready-to-use virtual environment.

    ```bash
    pixi install
    ```

---

## 🔬 Running Prediction and XML Generation

The analysis pipeline is contained in the `lmd_nb.py` script. This file is a **Jupyter Notebook** utilizing the **Jupytext percent** format, which allows it to be edited easily in any text editor while preserving its executable notebook structure.

### ⚙️ Configuration

Before running the pipeline, you must configure the parameters and channel mapping within the `lmd_nb.py` script:

* **Channel Configuration:** The fluorescence channel assignments must be set to match your input image data.
  * **Default Setup:** The notebook is currently defaulted to: **Marker (Ch 0), Autofluorescence (Ch 1), and DAPI (Ch 2)**. Adjust these channel indices (0-indexed) at the beginning of `lmd_nb.py` as needed.
* **Parameter Adjustment:** Parameters for specific processing steps can be adjusted directly before their corresponding notebook cells in `lmd_nb.py`.

### 🚀 Execution Command

Once configured, execute the pipeline using the following command:

```bash
pixi run python lmd_nb.py
```

---

### 📊 Example data
Example data for testing the *in silico* workflow is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17792442.svg)](https://doi.org/10.5281/zenodo.17792442)

---

### ✉️ Correspondence
[**Prof. Dr. Daniel R. Engel**](mailto:danielrobert.engel@uk-essen.de): Department of Immunodynamics, Institute of Experimental Immunology and Imaging, University Hospital Essen, Essen, Germany