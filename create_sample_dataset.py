#!/usr/bin/env python3
"""
Create Sample Dataset for Testing
================================

This script creates a sample dataset with placeholder files for testing the brain tumor detection system.
"""

import os
import numpy as np
from PIL import Image

def create_sample_image(filename, size=(128, 128), color=(100, 100, 100)):
    """Create a sample image file."""
    # Create a simple image
    img_array = np.full((size[0], size[1], 3), color, dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)
    return filename

def main():
    """Create sample dataset."""
    print("🧪 CREATING SAMPLE DATASET FOR TESTING")
    print("=" * 50)
    
    # Create directories
    data_dir = "data"
    yes_dir = os.path.join(data_dir, "yes")
    no_dir = os.path.join(data_dir, "no")
    
    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)
    
    # Create sample tumor images (darker, representing potential tumor areas)
    tumor_images = [
        ("sample_tumor_1.jpg", (128, 128), (50, 50, 50)),
        ("sample_tumor_2.jpg", (128, 128), (60, 60, 60)),
        ("sample_tumor_3.jpg", (128, 128), (40, 40, 40)),
        ("sample_tumor_4.jpg", (128, 128), (55, 55, 55)),
        ("sample_tumor_5.jpg", (128, 128), (45, 45, 45)),
    ]
    
    # Create sample normal images (lighter, representing normal brain tissue)
    normal_images = [
        ("sample_normal_1.jpg", (128, 128), (150, 150, 150)),
        ("sample_normal_2.jpg", (128, 128), (160, 160, 160)),
        ("sample_normal_3.jpg", (128, 128), (140, 140, 140)),
        ("sample_normal_4.jpg", (128, 128), (155, 155, 155)),
        ("sample_normal_5.jpg", (128, 128), (145, 145, 145)),
    ]
    
    print("Creating sample tumor images...")
    for filename, size, color in tumor_images:
        filepath = os.path.join(yes_dir, filename)
        create_sample_image(filepath, size, color)
        print(f"  ✅ Created: {filename}")
    
    print("Creating sample normal images...")
    for filename, size, color in normal_images:
        filepath = os.path.join(no_dir, filename)
        create_sample_image(filepath, size, color)
        print(f"  ✅ Created: {filename}")
    
    print(f"\n✅ Sample dataset created successfully!")
    print(f"   📁 Tumor images: {len(tumor_images)} (in {yes_dir})")
    print(f"   📁 Normal images: {len(normal_images)} (in {no_dir})")
    print("\n⚠️  NOTE: These are synthetic sample images for testing purposes.")
    print("   For real training, replace them with actual MRI images.")
    
    print("\n📊 NEXT STEPS:")
    print("1. Run preprocessing: python preprocess_data.py")
    print("2. Train the model: python train_model.py")
    print("3. Run the web app: python app.py")

if __name__ == "__main__":
    main()
