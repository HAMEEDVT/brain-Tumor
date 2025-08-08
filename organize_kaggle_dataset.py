#!/usr/bin/env python3
"""
Organize Kaggle Brain Tumor Dataset
===================================

This script helps organize the downloaded Kaggle dataset into the correct structure.
"""

import os
import shutil
import glob
from pathlib import Path

def organize_dataset(source_path, target_path="data"):
    """
    Organize the Kaggle dataset into the correct structure.
    
    Args:
        source_path (str): Path to the downloaded Kaggle dataset
        target_path (str): Target directory for organized data
    """
    print("🧠 ORGANIZING KAGGLE BRAIN TUMOR DATASET")
    print("=" * 50)
    
    # Create target directories
    yes_dir = os.path.join(target_path, "yes")
    no_dir = os.path.join(target_path, "no")
    
    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)
    
    print(f"Source path: {source_path}")
    print(f"Target path: {target_path}")
    
    # Check if source path exists
    if not os.path.exists(source_path):
        print(f"❌ Source path does not exist: {source_path}")
        print("Please provide the correct path to your downloaded dataset.")
        return False
    
    # Find all image files in the source directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(source_path, "**", ext), recursive=True))
        all_images.extend(glob.glob(os.path.join(source_path, ext), recursive=True))
    
    print(f"Found {len(all_images)} images in source directory")
    
    if len(all_images) == 0:
        print("❌ No images found in the source directory.")
        print("Please check the path and ensure the dataset contains image files.")
        return False
    
    # Organize images based on their location or filename
    moved_count = 0
    skipped_count = 0
    
    for img_path in all_images:
        filename = os.path.basename(img_path)
        dir_name = os.path.basename(os.path.dirname(img_path)).lower()
        
        # Determine target directory based on various indicators
        target_dir = None
        
        # Check directory name
        if any(keyword in dir_name for keyword in ['yes', 'tumor', 'positive', 'malignant']):
            target_dir = yes_dir
        elif any(keyword in dir_name for keyword in ['no', 'normal', 'negative', 'benign']):
            target_dir = no_dir
        
        # Check filename for indicators
        if target_dir is None:
            filename_lower = filename.lower()
            if any(keyword in filename_lower for keyword in ['tumor', 'positive', 'malignant', 'yes']):
                target_dir = yes_dir
            elif any(keyword in filename_lower for keyword in ['normal', 'negative', 'benign', 'no']):
                target_dir = no_dir
        
        # If still not determined, try to infer from the full path
        if target_dir is None:
            full_path_lower = img_path.lower()
            if any(keyword in full_path_lower for keyword in ['tumor', 'positive', 'malignant', 'yes']):
                target_dir = yes_dir
            elif any(keyword in full_path_lower for keyword in ['normal', 'negative', 'benign', 'no']):
                target_dir = no_dir
        
        # If still not determined, skip the file
        if target_dir is None:
            print(f"⚠️  Skipping {filename} - could not determine category")
            skipped_count += 1
            continue
        
        # Copy file to target directory
        target_file = os.path.join(target_dir, filename)
        
        # Handle duplicate filenames
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(target_file):
            target_file = os.path.join(target_dir, f"{base_name}_{counter}{ext}")
            counter += 1
        
        try:
            shutil.copy2(img_path, target_file)
            moved_count += 1
            print(f"✅ Moved: {filename} → {os.path.basename(target_dir)}/")
        except Exception as e:
            print(f"❌ Error copying {filename}: {e}")
            skipped_count += 1
    
    # Print summary
    print(f"\n📊 ORGANIZATION SUMMARY")
    print("=" * 30)
    print(f"✅ Successfully moved: {moved_count} images")
    print(f"⚠️  Skipped: {skipped_count} images")
    
    # Count final images
    yes_count = len([f for f in os.listdir(yes_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    no_count = len([f for f in os.listdir(no_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    
    print(f"📁 Tumor images (yes/): {yes_count}")
    print(f"📁 Normal images (no/): {no_count}")
    print(f"📁 Total images: {yes_count + no_count}")
    
    if yes_count > 0 and no_count > 0:
        print("\n✅ Dataset organized successfully!")
        print("You can now train the model with: python quick_train.py")
        return True
    else:
        print("\n❌ Dataset organization incomplete.")
        print("Please check the source directory structure and try again.")
        return False

def main():
    """Main function to organize the dataset."""
    print("Please provide the path to your downloaded Kaggle dataset.")
    print("Example paths:")
    print("- C:\\Users\\YourName\\Downloads\\brain-mri-images-for-brain-tumor-detection")
    print("- D:\\Datasets\\brain-tumor-detection")
    print("- /home/user/Downloads/brain-mri-images-for-brain-tumor-detection")
    
    source_path = input("\nEnter the path to your downloaded dataset: ").strip()
    
    if not source_path:
        print("❌ No path provided. Exiting.")
        return
    
    # Remove quotes if present
    source_path = source_path.strip('"\'')
    
    # Organize the dataset
    success = organize_dataset(source_path)
    
    if success:
        print("\n🎉 Dataset organization completed!")
        print("\n📋 NEXT STEPS:")
        print("1. Train the model: python quick_train.py")
        print("2. Run the web application: python app.py")
        print("3. Test with real MRI images")
    else:
        print("\n❌ Dataset organization failed.")
        print("Please check the path and try again.")

if __name__ == "__main__":
    main()
