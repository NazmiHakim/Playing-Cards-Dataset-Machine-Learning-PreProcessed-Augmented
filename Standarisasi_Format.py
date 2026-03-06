import cv2
import os
import numpy as np
from glob import glob

INPUT_DIR = r"C:\Users\ACER\Documents\ML_Tugas_1_Playing_Cards"
OUTPUT_DIR = r"C:\Users\ACER\Documents\CARD"
TARGET_SIZE = (224, 224) 

def standardize_dataset(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori output dibuat: {output_dir}")

    class_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    total_processed = 0

    print(f"Memulai standarisasi untuk {len(class_folders)} kelas...")

    for class_name in class_folders:
        input_class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        image_files = glob(os.path.join(input_class_path, '*.*')) 
        print(f"\nMemproses kelas '{class_name}' ({len(image_files)} gambar)...")

        for i, file_path in enumerate(image_files):
            try:
                img = cv2.imread(file_path, cv2.IMREAD_COLOR) 

                if img is None:
                    print(f"  [SKIP] Gagal memuat gambar: {file_path}")
                    continue

                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                img_normalized = img_resized.astype(np.float32) / 255.0

                file_name_base = os.path.splitext(os.path.basename(file_path))[0]
                output_file_name = f"{file_name_base}.jpg"
                output_file_path = os.path.join(output_class_path, output_file_name)

                img_to_save = (img_normalized * 255).astype(np.uint8)
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

standardize_dataset(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)