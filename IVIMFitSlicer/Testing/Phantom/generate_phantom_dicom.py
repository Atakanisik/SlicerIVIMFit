"""
Synthetic IVIM digital reference phantom generator (IVIMFitSlicer verification).

Produces a 100x100x10 DICOM series over b = [0,10,20,30,50,100,200,400,600,800] s/mm^2
with three nested tissue classes (bi-exponential forward model, S0 = 1000):
    healthy  : f = 0.10, D = 1.2e-3,  D* = 15e-3 mm^2/s   (outer region)
    tumour   : f = 0.25, D = 0.6e-3,  D* = 40e-3          (sphere r < 28)
    necrotic : f = 0.00, D = 2.5e-3,  D* = 0              (sphere r < 12)
Rician noise is added at a b = 0 signal-to-noise ratio SNR0 = S0/sigma = 60.
The generator is intentionally unseeded: each run yields an equivalent phantom
(identical ground-truth parameters, a fresh noise realization).

Requires: pip install pydicom numpy
"""

import os
import shutil
import numpy as np
# Requires pydicom (Colab: !pip install pydicom). Install once with: pip install pydicom
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
import pydicom.uid

# 1. Geometri ve IVIM Parametrelerinin Tanımlanması
grid_size = 100
num_slices = 10
b_values = [0, 10, 20, 30, 50, 100, 200, 400, 600, 800]
S0_base = 1000.0
SNR = 60.0
sigma = S0_base / SNR

output_dir = "Slicer_Ready_IVIM_DICOM"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Tüm seri için tek bir Study ve Series UID üretiyoruz (Slicer gruplaması için zorunlu)
study_uid = pydicom.uid.generate_uid()
series_uid = pydicom.uid.generate_uid()

instance_number = 1

# 3D Küresel Phantom Yapısı İçin Merkez Noktaları
center_x, center_y, center_z = grid_size // 2, grid_size // 2, num_slices // 2

print("DICOM dosyaları üretiliyor...")

# 2. 4D Döngü: b-değerleri ve Kesitler (Slices)
for b_idx, b_val in enumerate(b_values):
    for slice_idx in range(num_slices):

        # Matematiksel koordinat ızgarası
        Y, X = np.ogrid[:grid_size, :grid_size]
        # 3D Öklid mesafesi (Z eksenini kesit kalınlığı olan 5mm ile ölçeklendiriyoruz)
        distance_3d = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + ((slice_idx - center_z) * 2.5)**2)

        # IVIM Parametre Haritaları
        f_map = np.ones((grid_size, grid_size)) * 0.10
        D_map = np.ones((grid_size, grid_size)) * 1.2e-3
        Dstar_map = np.ones((grid_size, grid_size)) * 15.0e-3

        # Dokuların Geometrik Maskeleri (Küre Katmanları)
        tumor_mask = distance_3d < 28
        necrotic_mask = distance_3d < 12

        # Değer atamaları
        f_map[tumor_mask] = 0.25
        D_map[tumor_mask] = 0.6e-3
        Dstar_map[tumor_mask] = 40.0e-3

        f_map[necrotic_mask] = 0.0
        D_map[necrotic_mask] = 2.5e-3  # Nekrozda serbest sıvı/yüksek difüzyon
        Dstar_map[necrotic_mask] = 0.0

        # İleri Yönlü Sinyal Modellemesi (Bi-exponential)
        S_pure = S0_base * (f_map * np.exp(-b_val * Dstar_map) + (1.0 - f_map) * np.exp(-b_val * D_map))

        # Rician Gürültü Enjeksiyonu
        noise_real = np.random.normal(0, sigma, S_pure.shape)
        noise_imag = np.random.normal(0, sigma, S_pure.shape)
        S_noisy = np.sqrt((S_pure + noise_real)**2 + noise_imag**2)

        # Arka plan temizliği (Phantom dışı uzay sadece saf gürültü olsun)
        bg_mask = distance_3d > 42
        S_noisy[bg_mask] = np.sqrt(noise_real[bg_mask]**2 + noise_imag[bg_mask]**2)

        # DICOM için uint16 tipine dönüştürme
        image_matrix = np.clip(S_noisy, 0, 4000).astype(np.uint16)

        # --- DICOM DOSYASINI İNŞA ETME ---
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        filename = f"{output_dir}/IM_{instance_number:04d}_b{b_val}_sl{slice_idx}.dcm"
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Standart Kimlik Bilgileri
        ds.PatientName = "IVIM^Phantom"
        ds.PatientID = "PHANTOM_001"
        ds.Modality = "MR"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        # Seri ve Sıralama Bilgileri
        ds.SeriesNumber = 1001
        ds.InstanceNumber = instance_number

        # Matris Boyutları
        ds.Rows = grid_size
        ds.Columns = grid_size
        ds.PixelSpacing = [2.0, 2.0]       # 2mm x 2mm piksel boyutu
        ds.SliceThickness = 5.0            # 5mm kesit kalınlığı

        # Uzaysal Konumlandırma (3D Slicer'ın kesitleri üst üste yığmaması için kritik)
        z_pos = slice_idx * 5.0
        ds.ImagePositionPatient = [0.0, 0.0, z_pos]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        # --- KRİTİK DİFÜZYON METAVERİLERİ ---
        # Slicer'ın difüzyon algoritmalarını tetikleyen etiketler
        ds.add_new([0x0018, 0x9087], 'FD', b_val)              # b-value
        ds.add_new([0x0018, 0x9089], 'FD', [1.0, 0.0, 0.0])    # Dummy Yön Vektörü

        # Piksel Verisi Formatı
        ds.PixelData = image_matrix.tobytes()
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        ds.save_as(filename)
        instance_number += 1

# Klasörü indirmek için zipleyelim
shutil.make_archive(output_dir, 'zip', output_dir)
print(f"\nBaşarılı! 100 adet DICOM dosyası üretildi ve '{output_dir}.zip' olarak arşivlendi.")
