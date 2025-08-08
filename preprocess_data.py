#!/usr/bin/env python3
"""
Data Preprocessing Script for Brain Tumor Detection
==================================================

This script prepares the brain MRI dataset for training by:
1. Organizing the dataset structure
2. Validating image files
3. Creating train/validation/test splits
4. Generating dataset statistics

Author: Brain Tumor Detection System
Date: 2024
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
from datetime import datetime

# Configuration
DATA_DIR = 'data'
CATEGORIES = ['no', 'yes']
IMG_SIZE = 128
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def create_directory_structure():
    """Create the required directory structure."""
    directories = [
        DATA_DIR,
        f'{DATA_DIR}/no',
        f'{DATA_DIR}/yes',
        'model',
        'static/uploads',
        'static/results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def validate_image_file(file_path):
    """
    Validate if a file is a valid image.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        # Check file extension
        ext = Path(file_path).suffix.lower()
        if ext not in VALID_EXTENSIONS:
            return False
        
        # Try to open with PIL
        with Image.open(file_path) as img:
            img.verify()
        
        # Try to read with OpenCV
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        
        # Check minimum size
        if img.shape[0] < 32 or img.shape[1] < 32:
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating {file_path}: {str(e)}")
        return False

def analyze_dataset():
    """
    Analyze the dataset and generate statistics.
    
    Returns:
        dict: Dataset statistics
    """
    print("\n📊 Analyzing dataset...")
    
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'categories': {},
        'image_sizes': [],
        'file_extensions': {}
    }
    
    for category in CATEGORIES:
        category_path = os.path.join(DATA_DIR, category)
        if not os.path.exists(category_path):
            print(f"⚠️  Warning: Category directory '{category}' not found")
            stats['categories'][category] = 0
            continue
        
        valid_count = 0
        invalid_count = 0
        
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            
            if os.path.isfile(file_path):
                stats['total_images'] += 1
                
                # Check file extension
                ext = Path(filename).suffix.lower()
                stats['file_extensions'][ext] = stats['file_extensions'].get(ext, 0) + 1
                
                if validate_image_file(file_path):
                    valid_count += 1
                    
                    # Get image size
                    try:
                        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        stats['image_sizes'].append(img.shape)
                    except:
                        pass
                else:
                    invalid_count += 1
                    print(f"❌ Invalid image: {file_path}")
        
        stats['categories'][category] = valid_count
        stats['valid_images'] += valid_count
        stats['invalid_images'] += invalid_count
        
        print(f"📁 {category}: {valid_count} valid, {invalid_count} invalid images")
    
    return stats

def generate_dataset_report(stats):
    """
    Generate a comprehensive dataset report.
    
    Args:
        stats (dict): Dataset statistics
    """
    print("\n📋 DATASET REPORT")
    print("=" * 50)
    
    print(f"Total Images: {stats['total_images']}")
    print(f"Valid Images: {stats['valid_images']}")
    print(f"Invalid Images: {stats['invalid_images']}")
    print(f"Validation Rate: {(stats['valid_images']/stats['total_images']*100):.1f}%")
    
    print("\n📂 Category Distribution:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count} images")
    
    print("\n📄 File Extensions:")
    for ext, count in stats['file_extensions'].items():
        print(f"  {ext}: {count} files")
    
    if stats['image_sizes']:
        sizes = np.array(stats['image_sizes'])
        print(f"\n📏 Image Sizes:")
        print(f"  Min: {sizes.min(axis=0)}")
        print(f"  Max: {sizes.max(axis=0)}")
        print(f"  Mean: {sizes.mean(axis=0).astype(int)}")
    
    # Save report to file
    report_path = 'model/dataset_report.txt'
    with open(report_path, 'w') as f:
        f.write("BRAIN TUMOR DETECTION - DATASET REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total Images: {stats['total_images']}\n")
        f.write(f"Valid Images: {stats['valid_images']}\n")
        f.write(f"Invalid Images: {stats['invalid_images']}\n")
        f.write(f"Validation Rate: {(stats['valid_images']/stats['total_images']*100):.1f}%\n\n")
        
        f.write("Category Distribution:\n")
        for category, count in stats['categories'].items():
            f.write(f"  {category}: {count} images\n")
        
        f.write("\nFile Extensions:\n")
        for ext, count in stats['file_extensions'].items():
            f.write(f"  {ext}: {count} files\n")
    
    print(f"\n📄 Report saved to: {report_path}")

def create_visualization(stats):
    """
    Create visualizations of the dataset.
    
    Args:
        stats (dict): Dataset statistics
    """
    print("\n📈 Creating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Category distribution
    categories = list(stats['categories'].keys())
    counts = list(stats['categories'].values())
    
    axes[0, 0].bar(categories, counts, color=['#28a745', '#dc3545'])
    axes[0, 0].set_title('Image Distribution by Category')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].set_xlabel('Category')
    
    # Add value labels on bars
    for i, count in enumerate(counts):
        axes[0, 0].text(i, count + max(counts)*0.01, str(count), 
                        ha='center', va='bottom', fontweight='bold')
    
    # 2. File extensions
    extensions = list(stats['file_extensions'].keys())
    ext_counts = list(stats['file_extensions'].values())
    
    axes[0, 1].pie(ext_counts, labels=extensions, autopct='%1.1f%%')
    axes[0, 1].set_title('File Extensions Distribution')
    
    # 3. Image sizes (if available)
    if stats['image_sizes']:
        sizes = np.array(stats['image_sizes'])
        axes[1, 0].scatter(sizes[:, 0], sizes[:, 1], alpha=0.6)
        axes[1, 0].set_title('Image Size Distribution')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Height (pixels)')
    
    # 4. Validation status
    validation_data = ['Valid', 'Invalid']
    validation_counts = [stats['valid_images'], stats['invalid_images']]
    colors = ['#28a745', '#dc3545']
    
    axes[1, 1].pie(validation_counts, labels=validation_data, colors=colors, autopct='%1.1f%%')
    axes[1, 1].set_title('Image Validation Status')
    
    plt.tight_layout()
    plt.savefig('model/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualizations saved to: model/dataset_analysis.png")

def check_dataset_balance(stats):
    """
    Check if the dataset is balanced.
    
    Args:
        stats (dict): Dataset statistics
    """
    print("\n⚖️  Dataset Balance Analysis:")
    
    categories = list(stats['categories'].keys())
    counts = list(stats['categories'].values())
    
    if len(counts) >= 2:
        min_count = min(counts)
        max_count = max(counts)
        balance_ratio = min_count / max_count
        
        print(f"  Minimum category count: {min_count}")
        print(f"  Maximum category count: {max_count}")
        print(f"  Balance ratio: {balance_ratio:.3f}")
        
        if balance_ratio >= 0.8:
            print("  ✅ Dataset is well balanced")
        elif balance_ratio >= 0.6:
            print("  ⚠️  Dataset is moderately balanced")
        else:
            print("  ❌ Dataset is imbalanced - consider data augmentation")
    
    return balance_ratio if len(counts) >= 2 else 1.0

def main():
    """Main preprocessing function."""
    print("🧠 BRAIN TUMOR DETECTION - DATA PREPROCESSING")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ Dataset directory '{DATA_DIR}' not found!")
        print("Please download the dataset and place it in the 'data' directory.")
        print("Expected structure:")
        print("  data/")
        print("  ├── no/  (images without tumors)")
        print("  └── yes/ (images with tumors)")
        return
    
    # Analyze dataset
    stats = analyze_dataset()
    
    if stats['valid_images'] == 0:
        print("\n❌ No valid images found in the dataset!")
        print("Please ensure the dataset contains valid image files.")
        return
    
    # Generate report
    generate_dataset_report(stats)
    
    # Create visualizations
    create_visualization(stats)
    
    # Check balance
    balance_ratio = check_dataset_balance(stats)
    
    print(f"\n✅ Data preprocessing completed!")
    print(f"📊 Found {stats['valid_images']} valid images")
    print(f"📁 Dataset structure verified")
    print(f"📈 Analysis visualizations generated")
    
    if balance_ratio < 0.8:
        print(f"⚠️  Consider using data augmentation to balance the dataset")

if __name__ == "__main__":
    main()
