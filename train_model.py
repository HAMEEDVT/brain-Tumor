import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
IMG_SIZE = 128  # Increased for better feature extraction
DATA_DIR = 'data'
CATEGORIES = ['no', 'yes']  # No tumor first, then tumor
MODEL_PATH = 'model/brain_tumor_cnn.h5'
BATCH_SIZE = 32
EPOCHS = 50

def create_directories():
    """Create necessary directories for the project."""
    directories = ['model', 'static/uploads', 'static/results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_and_preprocess_data():
    """
    Load and preprocess the brain MRI dataset.
    
    Returns:
        tuple: (images, labels) - Preprocessed images and their labels
    """
    print("Loading and preprocessing data...")
    images = []
    labels = []
    
    for idx, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, category)
        if not os.path.exists(folder):
            print(f"Warning: {folder} does not exist. Please ensure the dataset is properly organized.")
            continue
            
        print(f"Processing {category} images...")
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize and normalize
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0  # Normalize to [0,1]
                    images.append(img)
                    labels.append(idx)
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Dataset loaded: {len(images)} images")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return images, labels

def build_enhanced_cnn():
    """
    Build an enhanced CNN model for brain tumor detection.
    
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_data_generators(X_train, y_train, X_val, y_val):
    """
    Create data generators for training with augmentation.
    
    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1.0)
    
    # Reshape data for generators
    X_train_reshaped = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_val_reshaped = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    train_generator = train_datagen.flow(X_train_reshaped, y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(X_val_reshaped, y_val, batch_size=BATCH_SIZE)
    
    return train_generator, val_generator

def plot_training_history(history):
    """Plot training history for accuracy and loss."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
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
    """Comprehensive model evaluation."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Basic metrics
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=['No Tumor', 'Tumor Present']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor Present'],
                yticklabels=['No Tumor', 'Tumor Present'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_classes, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('model/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'loss': test_loss,
        'auc': roc_auc
    }

def save_model_summary(model, metrics):
    """Save model summary and metrics to a text file."""
    with open('model/model_summary.txt', 'w') as f:
        f.write("BRAIN TUMOR DETECTION CNN MODEL SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image Size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n\n")
        
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
    print("BRAIN TUMOR DETECTION CNN TRAINING")
    print("="*50)
    
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build model
    print("\nBuilding CNN model...")
    model = build_enhanced_cnn()
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
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(X_val) // BATCH_SIZE,
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
    
    print(f"\nTraining completed! Model saved to {MODEL_PATH}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

if __name__ == "__main__":
    main()
