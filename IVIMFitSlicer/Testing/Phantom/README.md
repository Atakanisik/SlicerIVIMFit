# Synthetic IVIM Digital Reference Phantom

These scripts regenerate the digital reference phantom used to verify IVIMFitSlicer.
They are provided so the verification is fully reproducible; the (large) generated
DICOM/NRRD files are intentionally **not** committed.

## Files
| Script | Output | Purpose |
|---|---|---|
| `generate_phantom_dicom.py` | `Slicer_Ready_IVIM_DICOM/` (+ `.zip`) | 100×100×10 DICOM series, b = 0–800 s/mm² |
| `generate_segmentation.py` | `Slicer_Ready_Segmentation.nrrd` | ground-truth label map (tumour, necrotic) |

## Phantom definition
- Grid 100×100×10, voxel 2×2 mm in-plane, 5 mm slices.
- b-values: `0, 10, 20, 30, 50, 100, 200, 400, 600, 800` s/mm².
- Bi-exponential forward model, `S0 = 1000`.

| Tissue | f | D (mm²/s) | D* (mm²/s) | Region |
|---|---|---|---|---|
| Healthy | 0.10 | 1.2×10⁻³ | 15×10⁻³ | outer |
| Tumour | 0.25 | 0.6×10⁻³ | 40×10⁻³ | sphere r < 28 |
| Necrotic | 0.00 | 2.5×10⁻³ | 0 | sphere r < 12 |

- **Noise:** Rician, at a b = 0 signal-to-noise ratio `SNR0 = S0/σ = 60`.
- The generator is **unseeded**: each run yields an *equivalent* phantom (identical
  ground-truth parameters, a fresh noise realization). The ground-truth values — not a
  specific noise draw — define the phantom.

## ⚠️ Segmentation note
`generate_segmentation.py` produces **only the tumour (label 1) and necrotic (label 2)**
ROIs, which coincide exactly with the phantom geometry. The **healthy** and **background**
ROIs are created by the user in **3D Slicer → Segment Editor** (threshold the b = 0 image
to isolate signal-bearing tissue, then subtract the tumour and necrotic labels).

## How to run
```bash
pip install pydicom numpy
python generate_phantom_dicom.py     # writes the DICOM series
python generate_segmentation.py      # writes the label map
```
Then load the DICOM series and the label map into 3D Slicer and run IVIMFitSlicer.
