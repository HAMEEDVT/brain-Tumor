#!/usr/bin/env python3
"""
Quick Brain Tumor Detection Training
===================================

A minimal training script for small datasets.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Configuration
IMG_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "model/brain_tumor_model.h5"

def load_data():
    """Load the dataset."""
    print("Loading dataset...")
    
    data = []
    labels = []
    
    # Load no tumor images
    no_path = "data/no"
    if os.path.exists(no_path):
        for img_name in os.listdir(no_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_path = os.path.join(no_path, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        data.append(img)
                        labels.append(0)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Load yes tumor images
    yes_path = "data/yes"
    if os.path.exists(yes_path):
        for img_name in os.listdir(yes_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_path = os.path.join(yes_path, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        data.append(img)
                        labels.append(1)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    if len(data) == 0:
        return None, None
    
    # Convert to numpy arrays and normalize
    X = np.array(data) / 255.0
    y = np.array(labels)
    
    print(f"Loaded {len(X)} images")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y

def build_model():
    """Build a simple CNN model."""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main training function."""
    print("🧠 QUICK BRAIN TUMOR DETECTION TRAINING")
    print("="*50)
    
    # Create model directory
    os.makedirs("model", exist_ok=True)
    
    # Load data
    X, y = load_data()
    
    if X is None:
        print("❌ No images found. Please ensure the dataset is properly organized.")
        return
    
    # Reshape for CNN
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = to_categorical(y, num_classes=2)
    
    # For very small datasets, use all data for training
    print("⚠️  Using all data for training (small dataset mode)")
    X_train, y_train = X, y
    
    # Build and train model
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    print(f"\nTraining with {len(X_train)} samples...")
    print("This may take a few minutes...")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=1,  # Very small batch size
        verbose=1
    )
    
    # Save the model
    model.save(MODEL_PATH)
    
    print(f"\n✅ Training completed!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    print("\n📊 NEXT STEPS:")
    print("1. Run the web application: python app.py")
    print("2. Test with your MRI images")
    print("3. For better results, use a larger dataset")

if __name__ == "__main__":
    main()



