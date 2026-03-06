import albumentations as A
import cv2
import os
import numpy as np
from glob import glob

# --- Konfigurasi Direktori ---
# Ganti dengan direktori output dari FASE 3
INPUT_DIR = 'path/ke/dataset/standardized_224x224' 
# Direktori tempat gambar hasil augmentasi akan disimpan
OUTPUT_DIR = 'path/ke/dataset/augmented_5x'

# Ukuran gambar yang sudah distandardisasi
IMAGE_SIZE = 224

# Jumlah augmentasi yang ingin dihasilkan per gambar ASLI
# Jika TARGET_MULTIPLIER = 4, maka 1 gambar asli + 4 gambar augmentasi = 5x lipat data
TARGET_MULTIPLIER = 4 

# --- Definisikan Pipeline Augmentasi ---
# Teknik yang direkomendasikan untuk dataset kartu remi:

# 1. Rotasi (±10° hingga ±20°)
# 2. Horizontal Flip (Hindari Vertical Flip)
# 3. Penyesuaian Brightness dan Contrast
# 4. Zoom in/out ringan (±10%) & Random Crop (diwakili oleh RandomResizedCrop)

# Membuat pipeline augmentasi dengan albumentations
# Note: Augmentasi akan dilakukan pada gambar dalam rentang [0, 255] (uint8)
transform = A.Compose([
    # 1. Rotasi kecil (±20°)
    A.Rotate(limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    
    # 2. Horizontal Flip (diberikan probabilitas 50%)
    A.HorizontalFlip(p=0.5),
    
    # 3. Penyesuaian Brightness dan Contrast (batas ±20%)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    
    # 4. Zoom in/out ringan (ShiftScaleRotate: scale_limit=0.1 artinya ±10% scaling/zoom)
    A.ShiftScaleRotate(
        shift_limit=0.0, 
        scale_limit=0.1, 
        rotate_limit=0, # Rotasi sudah ditangani A.Rotate di atas
        interpolation=cv2.INTER_LINEAR, 
        border_mode=cv2.BORDER_REFLECT_101, 
        p=0.7
    ),
    
    # 5. Random Crop & Resize (Menggantikan zoom in/out yang lebih ekstrim, memastikan output tetap 224x224)
    # Ini akan mengambil porsi acak (crop) lalu resize kembali ke ukuran target.
    A.RandomResizedCrop(
        height=IMAGE_SIZE, 
        width=IMAGE_SIZE, 
        scale=(0.9, 1.0), # Crop/Zoom hanya 10%
        p=0.5
    ),

    # Tambahan: Blur ringan, untuk meningkatkan ketahanan model terhadap kualitas gambar
    A.Blur(blur_limit=3, p=0.3)
])

def augment_dataset(input_dir, output_dir, transform, multiplier):
    """
    Melakukan Fase 4: Augmentasi Data dan menyimpan hasilnya.

    Args:
        input_dir (str): Direktori dataset hasil Fase 3.
        output_dir (str): Direktori untuk menyimpan hasil augmentasi.
        transform (A.Compose): Pipeline augmentasi dari albumentations.
        multiplier (int): Jumlah augmentasi yang akan dihasilkan per gambar asli.
    """
    
    # Membuat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori output dibuat: {output_dir}")

    # Mendapatkan daftar folder per kelas
    class_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    total_original = 0
    total_augmented = 0

    print(f"Memulai augmentasi dengan multiplier {multiplier}x...")

    for class_name in class_folders:
        input_class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        image_files = glob(os.path.join(input_class_path, '*.jpg')) 
        
        current_original_count = len(image_files)
        total_original += current_original_count
        
        print(f"\nMemproses kelas '{class_name}' ({current_original_count} gambar asli)...")

        for i, file_path in enumerate(image_files):
            try:
                # Memuat gambar dalam format BGR (default OpenCV)
                img = cv2.imread(file_path)
                
                if img is None:
                    print(f"  [SKIP] Gagal memuat gambar: {file_path}")
                    continue
                
                # --- Menyimpan Gambar Asli ke Folder Output ---
                # Gambar asli (tanpa augmentasi) juga perlu dimasukkan ke folder augmented
                file_name_base = os.path.splitext(os.path.basename(file_path))[0]
                output_file_original = os.path.join(output_class_path, f"{file_name_base}_0.jpg")
                cv2.imwrite(output_file_original, img)
                total_augmented += 1

                # --- Menghasilkan Gambar Augmentasi (x 'multiplier') ---
                for j in range(1, multiplier + 1):
                    # Menerapkan transformasi
                    augmented = transform(image=img)
                    img_augmented = augmented['image']
                    
                    # Menyimpan gambar hasil augmentasi
                    output_file_augmented = os.path.join(output_class_path, f"{file_name_base}_{j}.jpg")
                    cv2.imwrite(output_file_augmented, img_augmented)
                    total_augmented += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Sudah memproses {i + 1} gambar asli di kelas '{class_name}'.")

            except Exception as e:
                print(f"  [ERROR] Terjadi kesalahan pada file {file_path}: {e}")
                continue

    print("\n--- AUGMENTASI SELESAI ---")
    print(f"Total gambar asli sebelum augmentasi: {total_original}")
    print(f"Target Multiplier: {multiplier + 1}x")
    print(f"Total gambar setelah augmentasi: {total_augmented}")
    print(f"Hasil disimpan di: {OUTPUT_DIR}")


# --- JALANKAN SCRIPT ---
# Ganti dengan path yang sesuai
augment_dataset(INPUT_DIR, OUTPUT_DIR, transform, TARGET_MULTIPLIER)