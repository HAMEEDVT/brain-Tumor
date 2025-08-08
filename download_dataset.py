#!/usr/bin/env python3
"""
Brain Tumor Dataset Downloader and Organizer
============================================

This script helps download and organize the brain tumor MRI dataset.
"""

import os
import zipfile
import requests
import shutil
from pathlib import Path
import sys

def download_file(url, filename):
    """Download a file from URL with progress bar."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            total_size = int(total_size)
            for data in response.iter_content(chunk_size=8192):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes")
                sys.stdout.flush()
    print(f"\n✅ Downloaded {filename}")

def extract_dataset(zip_path, extract_to):
    """Extract the dataset zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")

def organize_dataset(source_dir, target_dir):
    """Organize the dataset into yes/no directories."""
    print("Organizing dataset...")
    
    # Create target directories
    yes_dir = os.path.join(target_dir, 'yes')
    no_dir = os.path.join(target_dir, 'no')
    
    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)
    
    # Find and move files
    moved_count = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                
                # Determine if it's a tumor or non-tumor image based on directory name
                if 'yes' in root.lower() or 'tumor' in root.lower() or 'positive' in root.lower():
                    target_path = os.path.join(yes_dir, file)
                elif 'no' in root.lower() or 'non-tumor' in root.lower() or 'negative' in root.lower():
                    target_path = os.path.join(no_dir, file)
                else:
                    # If we can't determine, skip
                    continue
                
                # Avoid overwriting files with same name
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(os.path.dirname(target_path), f"{base}_{counter}{ext}")
                        counter += 1
                
                shutil.move(file_path, target_path)
                moved_count += 1
    
    print(f"✅ Moved {moved_count} images to organized directories")

def main():
    """Main function to download and organize the dataset."""
    print("🧠 BRAIN TUMOR DATASET DOWNLOADER")
    print("=" * 50)
    
    # Check if dataset already exists
    data_dir = "data"
    yes_dir = os.path.join(data_dir, "yes")
    no_dir = os.path.join(data_dir, "no")
    
    if os.path.exists(yes_dir) and os.path.exists(no_dir):
        yes_count = len([f for f in os.listdir(yes_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        no_count = len([f for f in os.listdir(no_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        
        if yes_count > 0 and no_count > 0:
            print(f"✅ Dataset already exists with {yes_count} tumor images and {no_count} non-tumor images")
            print("You can proceed to train the model or run the web application.")
            return
    
    print("📥 DATASET DOWNLOAD OPTIONS")
    print("1. Manual download (recommended)")
    print("2. Use sample dataset (for testing)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 40)
        print("1. Go to: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
        print("2. Download the dataset (requires free Kaggle account)")
        print("3. Extract the zip file to a temporary location")
        print("4. Run the preprocessing script: python preprocess_data.py")
        print("\nAfter downloading, place the images in the appropriate directories:")
        print(f"   - Tumor images → {yes_dir}")
        print(f"   - Non-tumor images → {no_dir}")
        
    elif choice == "2":
        print("\n🧪 CREATING SAMPLE DATASET FOR TESTING")
        print("This will create a small sample dataset for testing purposes.")
        
        # Create sample directories
        os.makedirs(yes_dir, exist_ok=True)
        os.makedirs(no_dir, exist_ok=True)
        
        # Create sample text files as placeholders
        sample_files = [
            (os.path.join(yes_dir, "sample_tumor_1.jpg"), "Sample tumor image 1"),
            (os.path.join(yes_dir, "sample_tumor_2.jpg"), "Sample tumor image 2"),
            (os.path.join(no_dir, "sample_normal_1.jpg"), "Sample normal image 1"),
            (os.path.join(no_dir, "sample_normal_2.jpg"), "Sample normal image 2"),
        ]
        
        for file_path, content in sample_files:
            with open(file_path, 'w') as f:
                f.write(f"# {content}\n# This is a placeholder file for testing\n# Replace with actual MRI images")
        
        print(f"✅ Created sample dataset with {len(sample_files)} placeholder files")
        print("⚠️  NOTE: These are placeholder files. Replace them with actual MRI images for real training.")
        
    elif choice == "3":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please run the script again.")
        return
    
    print("\n📊 NEXT STEPS")
    print("=" * 20)
    print("1. If you downloaded the dataset manually, run: python preprocess_data.py")
    print("2. Train the model: python train_model.py")
    print("3. Run the web application: python app.py")

if __name__ == "__main__":
    main()



