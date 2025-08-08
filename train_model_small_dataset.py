#!/usr/bin/env python3
"""
Brain Tumor Detection CNN Training - Small Dataset Version
=========================================================

This version is optimized for small datasets and testing purposes.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 2  # Smaller batch size for small dataset
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_PATH = "model/brain_tumor_model.h5"

def create_directories():
    """Create necessary directories."""
    os.makedirs("model", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print("Loading and preprocessing data...")
    
    data = []
    labels = []
    
    # Load no tumor images
    print("Processing no images...")
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
                        labels.append(0)  # 0 for no tumor
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Load yes tumor images
    print("Processing yes images...")
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
                        labels.append(1)  # 1 for tumor
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    if len(data) == 0:
        return [], []
    
    # Convert to numpy arrays
    X = np.array(data)
    y = np.array(labels)
    
    # Normalize pixel values
    X = X / 255.0
    
    print(f"Dataset loaded: {len(X)} images")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y

def build_simple_cnn():
    """Build a simple CNN model suitable for small datasets."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 classes: no tumor, tumor
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_data_generators(X_train, y_train, X_val, y_val):
    """Create data generators for training and validation."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Reshape data for generators
    X_train_reshaped = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_val_reshaped = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    train_generator = train_datagen.flow(
        X_train_reshaped, y_train,
        batch_size=BATCH_SIZE
    )
    
    val_generator = val_datagen.flow(
        X_val_reshaped, y_val,
        batch_size=BATCH_SIZE
    )
    
    return train_generator, val_generator

def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and generate metrics."""
    print("\nEvaluating model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('model/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor']))
    
    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'loss': test_loss,
        'auc': roc_auc
    }

def save_model_summary(model, metrics):
    """Save model summary and metrics to file."""
    with open('model/model_summary.txt', 'w') as f:
        f.write("BRAIN TUMOR DETECTION MODEL SUMMARY\n")
        f.write("="*40 + "\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-"*20 + "\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write("\nPERFORMANCE METRICS:\n")
        f.write("-"*20 + "\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Test Precision: {metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {metrics['recall']:.4f}\n")
        f.write(f"Test Loss: {metrics['loss']:.4f}\n")
        f.write(f"ROC AUC: {metrics['auc']:.4f}\n")

def main():
    """Main training function."""
    print("🧠 BRAIN TUMOR DETECTION CNN TRAINING (Small Dataset)")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("Error: No images found. Please ensure the dataset is properly organized in the 'data' directory.")
        return
    
    # Reshape data for CNN
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = to_categorical(y, num_classes=2)
    
    # For small datasets, use all data for training and create a small validation set
    if len(X) < 20:  # Very small dataset
        print("⚠️  Small dataset detected. Using all data for training with minimal validation split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        X_test = X_val  # Use validation as test set for small datasets
    else:
        # Normal split for larger datasets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build model
    print("\nBuilding CNN model...")
    model = build_simple_cnn()
    model.summary()
    
    # Create data generators
    train_generator, val_generator = create_data_generators(X_train, y_train, X_val, y_val)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH, 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15,  # More patience for small datasets
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=max(1, len(X_val) // BATCH_SIZE),
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model summary
    save_model_summary(model, metrics)
    
    print(f"\n✅ Training completed! Model saved to {MODEL_PATH}")
    if len(history.history['val_accuracy']) > 0:
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    print("\n📊 NEXT STEPS:")
    print("1. Run the web application: python app.py")
    print("2. Test with real MRI images")
    print("3. For production, replace sample dataset with real MRI images")

if __name__ == "__main__":
    main()



