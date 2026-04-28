# IVIMfitSlicer: Advanced Diffusion MRI Analysis for 3D Slicer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-3D%20Slicer-green.svg)
![Language](https://img.shields.io/badge/language-Python%203-yellow.svg)
![Status](https://img.shields.io/badge/status-stable-success.svg)

**IVIMfitSlicer** is a comprehensive 3D Slicer extension designed for the robust analysis of Intravoxel Incoherent Motion (IVIM) Diffusion-Weighted Imaging (DWI) data. 

It integrates the power of the `ivimfit` Python library directly into the clinical workflow of 3D Slicer, offering both **voxel-wise parametric mapping** and **ROI-averaged statistical modeling** with advanced visualization capabilities.

> **Research Grade:** Designed for high-precision fitting using multiple algorithms, including Segmented, Free, Tri-exponential, and Bayesian models.

---

## 🌟 Key Features

### 1. Multi-Model Fitting Algorithms
* **Mono-Exponential (ADC):** Standard Apparent Diffusion Coefficient calculation.
* **Bi-Exponential (IVIM):**
    * **Segmented Fit:** Robust two-step estimation (High-b first, then Low-b) to separate diffusion ($D$) from perfusion ($f, D^*$). *Default threshold: $b=200$.*
    * **Free Fit:** Non-linear least squares fitting for all parameters simultaneously.
* **Tri-Exponential:** Advanced compartmental analysis for separating fast, intermediate, and slow diffusion components ($f_{fast}, f_{interm}, f_{slow}$).
* **Bayesian Inference:** (Optional) MCMC-based parameter estimation for high-precision ROI statistics.

### 2. Hybrid Analysis & Visualization
* **Simultaneous Output:** Generates both 3D parametric maps and numerical ROI statistics in a single run.
* **Smart Visualization:**
    * **Auto-Layout:** Automatically switches Slicer layout to "Four-Up Quantitative" to show maps and fitting plots side-by-side.
    * **Interactive Plotting:** Plots the Raw Signal vs. Fitted Curve with $R^2$ values and calculated parameters directly in the Slicer interface.
    * **Auto-Scaling:** Automatically scales diffusion maps ($\times 10^6$) and fractions ($\times 100$) to ensure optimal visibility and contrast.

### 3. Workflow Efficiency
* **Sequence & MultiVolume Support:** Seamlessly handles 4D datasets, including Sequence Nodes and MultiVolume nodes.
* **Standard Clinical Preset:** Includes a dedicated **"Load 10-Point Preset"** button to instantly apply standard b-values (`0, 50, 100, 150, 200, 300, 400, 500, 700, 900`) for rapid analysis setup.

---

## 🛠️ Installation

### Option 1: Via Slicer Extensions Manager
1.  Open 3D Slicer.
2.  Go to the **Extensions Manager**.
3.  Search for `IVIMfitSlicer` and install.

### Option 2: Manual Installation (Developer Mode)
1.  Clone or download this repository.
2.  Open 3D Slicer.
3.  Go to **Developer Tools** -> **Extension Wizard**.
4.  Click **Select Extension** and navigate to the `IVIMfitSlicer` folder.
5.  Ensure the module is checked and accept the restart if prompted.

---

---

## ⚡ Advanced Setup for MCMC (Bayesian Mode) Strongly Recommended
*Optional: Only required for the "Bayesian " algorithm.*

The **Bayesian ** mode performs extensive Markov Chain Monte Carlo (MCMC) sampling (10,000 draws, 3 chains) to provide highly accurate parameter estimations and Gelman-Rubin (R-hat) convergence statistics. 

**The Computational Bottleneck:**
Computing massive MCMC chains in pure Python can take hours or even a full day. To achieve high performance, the underlying `PyMC`/`PyTensor` engine attempts to compile the mathematical models into C++ code. However, 3D Slicer's embedded Python environment is stripped down to save space and lacks the necessary C++ headers (`include`) and linking libraries (`libs`).

**The Solution (C++ Acceleration):**
If you want to run the Quality mode in minutes rather than days, you must manually link a standard Python environment to your 3D Slicer installation to enable C++ compilation. If you skip this setup, the module will fallback to pure Python (expect extremely long computation times).

### Prerequisites (Windows)
1. A C++ Compiler (e.g., `g++` via MSYS2/MinGW) added to your system PATH.
2. A standard local Python installation that matches Slicer's Python version (e.g., Python 3.9 or 3.12).

### Step-by-Step Linking Guide
1. **Locate Slicer's Python Directory:**
   Navigate to your Slicer installation path, typically:
   `C:\Users\<YourUsername>\AppData\Local\slicer.org\3D Slicer <Version>\lib\Python`
2. **Fix the Linking Libraries (`libs`):**
   * Inside Slicer's Python folder, create a new empty folder named `libs`.
   * Go to your system's standard Python installation folder (e.g., `C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python312\libs`).
   * Copy the library file (e.g., `python3.lib` and/or `python312.lib`) and paste it into the new `libs` folder inside Slicer.
3. **Fix the Headers (`include`):**
   * Go to your system's standard Python folder and copy the entire `include` folder (which contains `Python.h`).
   * Paste it directly into Slicer's Python directory (next to the `libs` folder you just created).
4. **Restart 3D Slicer as Administrative Mode:**
   Run the Bayesian  mode. The system will now detect `g++` and the header files, compiling the model in seconds and unlocking maximum hardware performance.

---

**Strategic Design Choice regarding Parametric Mapping:**
Computing Markov Chain Monte Carlo (MCMC) simulations on a voxel-by-voxel basis for 3D parametric mapping is computationally prohibitive (potentially taking days for a single volume). 

To maximize workflow efficiency without sacrificing statistical rigor, **IVIMfitSlicer** employs a hybrid approach:
 * **3D Parametric Maps** are generated rapidly using the robust **Segmented algorithm** for **Bayesian** selections, providing immediate visual feedback and spatial distribution context.
 * **ROI-Averaged Statistics**, which are critical for scientific conclusions, are computed using the selected **Bayesian Inference** model to provide the highest parameter precision and R-hat convergence values.

---
## 🚀 Usage Workflow

### 1. Load Data
Import your DWI dataset (DICOM, NRRD, or Sequence) into 3D Slicer.

### 2. Define Region of Interest (ROI)
* Go to the **Segment Editor** module.
* Create a segmentation mask covering the tissue of interest (e.g., Tumor, Liver, Kidney).
* *Note: Analysis is strictly ROI-based to ensure computational efficiency and high SNR.*

### 3. Configure IVIMfitSlicer
* Switch to the **IVIMfitSlicer** module (under Diffusion category).
* **Input Image:** Select your DWI sequence.
* **ROI Mask:** Select the segmentation you created.
* **B-Values:** Enter b-values manually or click **"Load 10-Point Preset"** for quick configuration.
* **Algorithm:** Choose your desired model (e.g., `Bi-Exponential (Segmented)`).

### 4. Run Analysis
* Click **"Run Analysis & Plot"**.
* The module will:
    1.  Extract data from the ROI.
    2.  Perform voxel-wise fitting (generating maps).
    3.  Perform ROI-averaged fitting (generating statistics).
* You can change view layout and obsreve fitting curve with PlotCharts.
* You can change shown map by selecting in "Data" section.    
---
### 🖼️ Workflow Gallery

<p align="center">
  <strong>Step 1: Configuration (Main Window)</strong><br>
  <img src="Screenshots/MainInterface.png" alt="Main Interface Before Analysis" width="800" style="margin-bottom: 20px;">
  <br>
  <em>Setting up inputs, ROI mask, and selecting the preset with Bi-Exponential Segmented model.</em>
</p>

<p align="center">
  <strong>Step 2: Analysis Results (Maps & Plots)</strong><br>
  <img src="Screenshots/Map_Plot_Screen.png" alt="Analysis Results View" width="800">
  <br>
  <em> Generated Diffusion (D) Map and the ROI fitting curve with calculated parameters .</em>
</p>
---
---

## 🧮 Methodological Background

**IVIM Bi-Exponential Model:**
$$\frac{S(b)}{S_0} = f \cdot e^{-b D^*} + (1-f) \cdot e^{-b D}$$
* **$D$**: True diffusion coefficient (Slow component).
* **$D^*$**: Pseudo-diffusion coefficient (Fast/Perfusion component).
* **$f$**: Perfusion fraction.

**Tri-Exponential Model:**
$$\frac{S(b)}{S_0} = f_{fast} \cdot e^{-b D^*_{fast}} + f_{interm} \cdot e^{-b D^*_{interm}} + f_{slow} \cdot e^{-b D}$$
Useful for tissues with three distinct diffusion compartments.

---

## 🤝 Dependencies

This extension automatically manages the following Python dependencies:
* `ivimfit` (Core calculation engine)
* `numpy`, `scipy`
* `pydicom`

---

## 📄 Sample Data

You can find Sample Data on Slicer's Sample Data Module

---

## 📄 License & Citation

**Author:** Atakan Isik (Baskent University)  
**License:** MIT License.
If you use this software in your research, please cite the associated repository or publication[10.36948/ijfmr.2025.v07i05.56036](https://doi.org/10.36948/ijfmr.2025.v07i05.56036).

---


*Developed for the Slicer Community.*


