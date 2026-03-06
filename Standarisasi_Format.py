import cv2
import os
import numpy as np
from glob import glob

# --- Konfigurasi Direktori ---
# Ganti 'path/ke/dataset/mentah' dengan lokasi folder utama yang berisi subfolder kelas (spade, heart, club, diamond)
INPUT_DIR = 'path/ke/dataset/mentah'
# Direktori tempat gambar hasil standarisasi akan disimpan
OUTPUT_DIR = 'path/ke/dataset/standardized_224x224'

# Ukuran target
TARGET_SIZE = (224, 224) 

def standardize_dataset(input_dir, output_dir, target_size):
    """
    Melakukan Fase 3: Standarisasi Format (Resize, Konversi Format JPG, Normalisasi).

    Args:
        input_dir (str): Direktori dataset mentah.
        output_dir (str): Direktori untuk menyimpan hasil.
        target_size (tuple): Ukuran target (width, height).
    """
    
    # Membuat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori output dibuat: {output_dir}")

    # Mengambil daftar folder per kelas (spade, heart, club, diamond, dll.)
    class_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    total_processed = 0

    print(f"Memulai standarisasi untuk {len(class_folders)} kelas...")

    for class_name in class_folders:
        input_class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        
        # Membuat sub-direktori kelas di output
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        # Mencari semua file gambar di dalam folder kelas
        # Menggunakan * untuk mencakup format seperti jpg, png, jpeg
        image_files = glob(os.path.join(input_class_path, '*.*')) 
        
        print(f"\nMemproses kelas '{class_name}' ({len(image_files)} gambar)...")

        for i, file_path in enumerate(image_files):
            try:
                # 1. Memuat gambar
                # cv2.IMREAD_COLOR memastikan gambar dimuat sebagai RGB (3 channel)
                img = cv2.imread(file_path, cv2.IMREAD_COLOR) 

                if img is None:
                    print(f"  [SKIP] Gagal memuat gambar: {file_path}")
                    continue

                # 2. Ukuran (Resize) - ke 224x224 piksel
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # 3. Color channel - cv2.imread(cv2.IMREAD_COLOR) sudah memastikan RGB (3 channel)

                # 4. Normalisasi piksel - dari [0, 255] menjadi [0, 1]
                # Konversi ke float32 agar bisa dibagi
                img_normalized = img_resized.astype(np.float32) / 255.0

                # Mendapatkan nama file baru dengan format JPG
                # cv2.imwrite akan otomatis menyimpan dalam format yang diminta dari ekstensi
                file_name_base = os.path.splitext(os.path.basename(file_path))[0]
                output_file_name = f"{file_name_base}.jpg"
                output_file_path = os.path.join(output_class_path, output_file_name)

                # Menyimpan gambar hasil normalisasi ke disk
                # Untuk menyimpan gambar float [0, 1] dengan cv2.imwrite, 
                # kita harus mengalikannya kembali ke [0, 255] dan mengkonversi ke uint8.
                # Normalisasi [0, 1] *biasanya* hanya dilakukan di memori sebelum masuk ke model training,
                # tetapi untuk tujuan penyimpanan output *sementara* ini, kita simpan kembali ke [0, 255]
                # dalam format JPG yang standar, dan normalisasi [0, 1] dilakukan ulang
                # nanti di script data loader/generator sebelum model dilatih.
                # Namun, jika Anda *memang* ingin menyimpan file dalam kondisi ter-normalisasi, 
                # Anda mungkin perlu format file yang mendukung nilai float (misalnya .npy atau .png 16-bit),
                # tetapi karena permintaan adalah JPG, kita ikuti praktik umum.

                # --- PRINSIP UMUM: Normalisasi [0, 1] dilakukan saat LOAD data untuk TRAINING ---
                # Untuk output penyimpanan fisik (JPG), kita simpan gambar 224x224, 3-channel, format JPG.
                
                img_to_save = (img_normalized * 255).astype(np.uint8)
                
                # Format file - Konversi semua gambar ke format JPG (dicapai oleh ekstensi .jpg)
                cv2.imwrite(output_file_path, img_to_save)
                
                total_processed += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Sudah memproses {i + 1} gambar di kelas '{class_name}'.")

            except Exception as e:
                print(f"  [ERROR] Terjadi kesalahan pada file {file_path}: {e}")
                continue

    print("\n--- STANDARISASI SELESAI ---")
    print(f"Total gambar yang diproses dan disimpan: {total_processed}")
    print(f"Hasil disimpan di: {OUTPUT_DIR}")


# --- JALANKAN SCRIPT ---
# Ganti dengan path yang sesuai
standardize_dataset(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)