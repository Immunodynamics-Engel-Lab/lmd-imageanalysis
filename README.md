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
    git clone [https://github.com/YourUsername/lmd-imageanalysis-pipeline.git](https://github.com/YourUsername/lmd-imageanalysis-pipeline.git)
    cd lmd-imageanalysis-pipeline
    ```

2.  **Initialize the Pixi Environment:**

    Pixi reads the required dependencies from the `pixi.toml` file and creates a ready-to-use virtual environment.

    ```bash
    pixi install
    ```

---

## ▶️ Execution

All execution is performed using the `pixi run` command to ensure the correct environment and dependencies are used.

### Running Prediction on Images

The pipeline is in the `lmd_nb.py` script. This script is a jupyter notebook using the jupytext percent format that can be adapted in any text editor and run using the execution command below.

```bash
pixi run python lmd_nb.py
```