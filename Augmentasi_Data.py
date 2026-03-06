import albumentations as A
import cv2
import os
import numpy as np
from glob import glob

INPUT_DIR = r"C:\Users\ACER\Documents\CARD"
OUTPUT_DIR = r"C:\Users\ACER\Documents\CARD2"
IMAGE_SIZE = 224
TARGET_MULTIPLIER = 4 

transform = A.Compose([
    A.Rotate(limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=None,
        rotate=0,
        interpolation=cv2.INTER_LINEAR, 
        mode=cv2.BORDER_REFLECT_101,
        p=0.7
    ),
    A.RandomResizedCrop(
        size=(IMAGE_SIZE, IMAGE_SIZE),
        scale=(0.9, 1.0),
        p=0.5
    ),
    A.Blur(blur_limit=3, p=0.3)
])

def augment_dataset(input_dir, output_dir, transform, multiplier):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori output dibuat: {output_dir}")

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
                img = cv2.imread(file_path)
                
                if img is None:
                    print(f"  [SKIP] Gagal memuat gambar: {file_path}")
                    continue
                
                file_name_base = os.path.splitext(os.path.basename(file_path))[0]
                output_file_original = os.path.join(output_class_path, f"{file_name_base}_0.jpg")
                cv2.imwrite(output_file_original, img)
                total_augmented += 1

                for j in range(1, multiplier + 1):
                    augmented = transform(image=img)
                    img_augmented = augmented['image']
                    
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

augment_dataset(INPUT_DIR, OUTPUT_DIR, transform, TARGET_MULTIPLIER)