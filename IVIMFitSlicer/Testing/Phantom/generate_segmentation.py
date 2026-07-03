"""
Matched ground-truth segmentation for the synthetic IVIM phantom
(see generate_phantom_dicom.py).

Generates an NRRD label map that is geometrically identical to the phantom, with:
    label 1 = tumour   (sphere r < 28)
    label 2 = necrotic (sphere r < 12)

NOTE: only the tumour and necrotic ROIs are produced here. The **healthy** and
**background** ROIs are created by the user in 3D Slicer's Segment Editor
(e.g., threshold the b = 0 image to isolate signal-bearing tissue, then subtract
the tumour and necrotic labels). This mirrors the analysis workflow in the paper.

Requires: pip install numpy
"""

import numpy as np

# 1. Geometrik Tanımlamalar (Sizin kodunuzla birebir aynı)
grid_size = 100
num_slices = 10
center_x, center_y, center_z = grid_size // 2, grid_size // 2, num_slices // 2

# Sizin döngünüzün ürettiği (Y, X, Z) yapısında boş bir matris oluşturalım
labelmap_3d = np.zeros((grid_size, grid_size, num_slices), dtype=np.int16)

print("Segmentasyon maskesi hesaplanıyor...")

# 2. Maskelerin Sizin Kodunuzdaki Geometriyle Birebir Aynı Oluşturulması
for slice_idx in range(num_slices):
    Y, X = np.ogrid[:grid_size, :grid_size]
    # Sizin kodunuzdaki 3D Öklid mesafesi formülünün aynısı
    distance_3d = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + ((slice_idx - center_z) * 2.5)**2)

    tumor_mask = distance_3d < 28
    necrotic_mask = distance_3d < 12

    # Etiket değerleri (1: Tümör, 2: Nekroz)
    labelmap_3d[tumor_mask, slice_idx] = 1
    labelmap_3d[necrotic_mask, slice_idx] = 2

# --- KRİTİK BELLEK DÜZELTMESİ (Görüntünün Bozulmasını Engelleyen Kısım) ---
# Sizin kodunuzda matris (Y, X, Z) düzenindedir. NRRD formatı ise en hızlı değişen eksenin
# X (sizes: 100 100 10) olmasını bekler. Numpy'da en sağdaki eksen en hızlı değiştiği için
# matrisi (Z, Y, X) düzenine transpoze ediyoruz.
corrected_labelmap = np.transpose(labelmap_3d, (2, 0, 1)).copy()

# Sizin DICOM verilerinizin orijini [0,0,0], Spacing=[2,2] ve Kalınlık=5 değerleriyle tam çakışan başlık
nrrd_header = (
    "NRRD0004\n"
    "type: short\n"
    "dimension: 3\n"
    "space: left-posterior-superior\n"
    "sizes: 100 100 10\n"
    "space directions: (2,0,0) (0,2,0) (0,0,5)\n"
    "kinds: domain domain domain\n"
    "endian: little\n"
    "encoding: raw\n"
    "space origin: (0,0,0)\n\n"
)

output_filename = "Slicer_Ready_Segmentation.nrrd"
with open(output_filename, "wb") as f:
    f.write(nrrd_header.encode('utf-8'))
    f.write(corrected_labelmap.tobytes())

print(f"\nBaşarılı! Maske dosyası '{output_filename}' olarak üretildi.")
