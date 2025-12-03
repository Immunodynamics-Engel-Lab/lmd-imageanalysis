# Segmentation and Reference Point Detection for Laser Capture Microdissection (LMD)

## Project Summary

This repository contains the code for a [Cellpose-SAM](https://github.com/MouseLand/cellpose) & [pyLMD](https://github.com/MannLabs/py-lmd) project dedicated to automating **cell boundary and reference point detection** in microscopic images used for **Laser Capture Microdissection (LMD)**.

The primary function of this repository is to identify the boundaries of target cells and detect precise reference points from laser engraved 'T' structures and is designed to enhance the reproducibility of the LMD workflow.

---

## Getting Started

### 📋 Prerequisites

This project exclusively uses the [**Pixi**](https://pixi.sh/latest/python/tutorial/) package manager to guarantee a reliable and isolated Python environment. [**Install Instructions**](https://prefix-dev.github.io/pixi/main/install.html)

### 💻 Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourUsername/lmd-imageanalysis-pipeline.git
    cd lmd-imageanalysis-pipeline
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