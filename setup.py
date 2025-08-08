#!/usr/bin/env python3
"""
Setup Script for Brain Tumor Detection System
============================================

This script helps users set up the brain tumor detection project by:
1. Creating necessary directories
2. Checking Python environment
3. Verifying dependencies
4. Providing setup instructions

Author: Brain Tumor Detection System
Date: 2024
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def print_banner():
    """Print the project banner."""
    print("🧠 BRAIN TUMOR DETECTION SYSTEM")
    print("=" * 50)
    print("Setting up your environment...")
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   This project requires Python 3.8 or higher")
        print("   Please upgrade your Python installation")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    """Create necessary project directories."""
    print("📁 Creating project directories...")
    
    directories = [
        'data',
        'data/no',
        'data/yes',
        'model',
        'static',
        'static/uploads',
        'static/results',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    print()

def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'tensorflow',
        'keras',
        'flask',
        'numpy',
        'opencv-python',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pillow',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_dataset():
    """Check if the dataset is properly organized."""
    print("📊 Checking dataset...")
    
    data_dir = Path('data')
    no_dir = data_dir / 'no'
    yes_dir = data_dir / 'yes'
    
    if not data_dir.exists():
        print("   ❌ Data directory not found")
        print("   Please create the 'data' directory")
        return False
    
    if not no_dir.exists():
        print("   ❌ 'data/no' directory not found")
        print("   Please create the 'data/no' directory for images without tumors")
        return False
    
    if not yes_dir.exists():
        print("   ❌ 'data/yes' directory not found")
        print("   Please create the 'data/yes' directory for images with tumors")
        return False
    
    # Count images in each directory
    no_images = len(list(no_dir.glob('*.jpg')) + list(no_dir.glob('*.png')) + list(no_dir.glob('*.jpeg')))
    yes_images = len(list(yes_dir.glob('*.jpg')) + list(yes_dir.glob('*.png')) + list(yes_dir.glob('*.jpeg')))
    
    print(f"   📁 No tumor images: {no_images}")
    print(f"   📁 Tumor images: {yes_images}")
    
    if no_images == 0 and yes_images == 0:
        print("   ⚠️  No images found in dataset directories")
        print("   Please download the dataset and place images in the appropriate directories")
        return False
    
    print("✅ Dataset structure is correct")
    return True

def check_model():
    """Check if trained model exists."""
    print("🧠 Checking model...")
    
    model_path = Path('model/brain_tumor_cnn.h5')
    
    if model_path.exists():
        print("   ✅ Trained model found")
        return True
    else:
        print("   ❌ Trained model not found")
        print("   Run: python train_model.py")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎯 NEXT STEPS")
    print("=" * 30)
    
    print("1. 📊 Preprocess your dataset:")
    print("   python preprocess_data.py")
    print()
    
    print("2. 🤖 Train the model:")
    print("   python train_model.py")
    print()
    
    print("3. 🔍 Evaluate the model:")
    print("   python evaluate_model.py")
    print()
    
    print("4. 🌐 Run the web application:")
    print("   python app.py")
    print()
    
    print("5. 📖 Read the documentation:")
    print("   See README.md for detailed instructions")
    print()

def print_dataset_instructions():
    """Print dataset setup instructions."""
    print("\n📁 DATASET SETUP INSTRUCTIONS")
    print("=" * 40)
    print("1. Download the dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
    print()
    print("2. Extract the dataset and organize as follows:")
    print("   data/")
    print("   ├── no/     (images without tumors)")
    print("   └── yes/    (images with tumors)")
    print()
    print("3. Run the preprocessing script:")
    print("   python preprocess_data.py")
    print()

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    print()
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    print()
    
    # Check dataset
    dataset_ok = check_dataset()
    print()
    
    # Check model
    model_ok = check_model()
    print()
    
    # Print summary
    print("📋 SETUP SUMMARY")
    print("=" * 20)
    print(f"Python Environment: {'✅' if check_python_version() else '❌'}")
    print(f"Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"Dataset: {'✅' if dataset_ok else '❌'}")
    print(f"Trained Model: {'✅' if model_ok else '❌'}")
    print()
    
    if not deps_ok:
        print("⚠️  Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print()
    
    if not dataset_ok:
        print_dataset_instructions()
    
    if deps_ok and dataset_ok and not model_ok:
        print_next_steps()
    elif deps_ok and dataset_ok and model_ok:
        print("🎉 Setup complete! You can now run the web application:")
        print("   python app.py")
    else:
        print("⚠️  Please complete the setup steps above before proceeding")

if __name__ == "__main__":
    main()
