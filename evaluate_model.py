#!/usr/bin/env python3
"""
Model Evaluation Script for Brain Tumor Detection
================================================

This script provides comprehensive evaluation of the trained CNN model:
1. Load trained model
2. Evaluate on test data
3. Generate detailed performance metrics
4. Create visualizations
5. Generate evaluation report

Author: Brain Tumor Detection System
Date: 2024
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Configuration
MODEL_PATH = 'model/brain_tumor_cnn.h5'
DATA_DIR = 'data'
CATEGORIES = ['no', 'yes']
IMG_SIZE = 128

def load_test_data():
    """
    Load and preprocess test data.
    
    Returns:
        tuple: (X_test, y_test) - Test images and labels
    """
    print("Loading test data...")
    
    images = []
    labels = []
    
    for idx, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, category)
        if not os.path.exists(folder):
            print(f"Warning: {folder} does not exist")
            continue
            
        print(f"Processing {category} images...")
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    images.append(img)
                    labels.append(idx)
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert to categorical
    labels = to_categorical(labels, num_classes=2)
    
    print(f"Loaded {len(images)} test images")
    return images.reshape(-1, IMG_SIZE, IMG_SIZE, 1), labels

def load_model_safely():
    """
    Load the trained model with error handling.
    
    Returns:
        model: Loaded Keras model or None
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return None
    
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Prediction probabilities
        
    Returns:
        dict: Performance metrics
    """
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    pr_auc = average_precision_score(y_true, y_pred_proba[:, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'npv': npv,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        },
        'pr_curve': {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist()
        }
    }

def create_evaluation_plots(metrics, y_true, y_pred, y_pred_proba):
    """
    Create comprehensive evaluation visualizations.
    
    Args:
        metrics (dict): Performance metrics
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Prediction probabilities
    """
    print("Creating evaluation plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor Present'],
                yticklabels=['No Tumor', 'Tumor Present'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr = np.array(metrics['roc_curve']['fpr'])
    tpr = np.array(metrics['roc_curve']['tpr'])
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    # 3. Precision-Recall Curve
    precision_curve = np.array(metrics['pr_curve']['precision'])
    recall_curve = np.array(metrics['pr_curve']['recall'])
    axes[0, 2].plot(recall_curve, precision_curve, color='green', lw=2,
                     label=f'PR curve (AP = {metrics["pr_auc"]:.3f})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 4. Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_values = [
        metrics['accuracy'], metrics['precision'], metrics['recall'],
        metrics['f1_score'], metrics['specificity']
    ]
    
    bars = axes[1, 0].bar(metric_names, metric_values, color=['#28a745', '#007bff', '#ffc107', '#dc3545', '#6f42c1'])
    axes[1, 0].set_title('Performance Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Prediction Distribution
    axes[1, 1].hist(y_pred_proba[:, 1], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance Summary
    summary_text = f"""
    Model Performance Summary
    
    Accuracy: {metrics['accuracy']:.3f}
    Precision: {metrics['precision']:.3f}
    Recall: {metrics['recall']:.3f}
    F1-Score: {metrics['f1_score']:.3f}
    Specificity: {metrics['specificity']:.3f}
    ROC AUC: {metrics['roc_auc']:.3f}
    PR AUC: {metrics['pr_auc']:.3f}
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_title('Performance Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('model/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Evaluation plots saved to: model/evaluation_results.png")

def generate_evaluation_report(metrics, model_info):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        metrics (dict): Performance metrics
        model_info (dict): Model information
    """
    print("Generating evaluation report...")
    
    report_path = 'model/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("BRAIN TUMOR DETECTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 20 + "\n")
        if model_info:
            f.write(f"Input Shape: {model_info['input_shape']}\n")
            f.write(f"Output Shape: {model_info['output_shape']}\n")
            f.write(f"Total Parameters: {model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
            f.write(f"Non-trainable Parameters: {model_info['non_trainable_params']:,}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall (Sensitivity): {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"Negative Predictive Value: {metrics['npv']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 20 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"True Negatives: {cm['tn']}\n")
        f.write(f"False Positives: {cm['fp']}\n")
        f.write(f"False Negatives: {cm['fn']}\n")
        f.write(f"True Positives: {cm['tp']}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-" * 20 + "\n")
        if metrics['accuracy'] >= 0.9:
            f.write("✅ Excellent accuracy achieved\n")
        elif metrics['accuracy'] >= 0.8:
            f.write("✅ Good accuracy achieved\n")
        elif metrics['accuracy'] >= 0.7:
            f.write("⚠️  Moderate accuracy - consider model improvements\n")
        else:
            f.write("❌ Low accuracy - model needs significant improvements\n")
        
        if metrics['roc_auc'] >= 0.9:
            f.write("✅ Excellent discriminative ability\n")
        elif metrics['roc_auc'] >= 0.8:
            f.write("✅ Good discriminative ability\n")
        else:
            f.write("⚠️  Limited discriminative ability\n")
    
    print(f"📄 Evaluation report saved to: {report_path}")

def save_metrics_json(metrics):
    """
    Save metrics to JSON file for programmatic access.
    
    Args:
        metrics (dict): Performance metrics
    """
    json_path = 'model/evaluation_metrics.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            json_metrics[key] = value.tolist()
        else:
            json_metrics[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"📊 Metrics saved to JSON: {json_path}")

def main():
    """Main evaluation function."""
    print("🧠 BRAIN TUMOR DETECTION - MODEL EVALUATION")
    print("=" * 50)
    
    # Load model
    model = load_model_safely()
    if model is None:
        return
    
    # Get model information
    model_info = {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    }
    
    # Load test data
    X_test, y_test = load_test_data()
    
    if len(X_test) == 0:
        print("❌ No test data found!")
        return
    
    print(f"Evaluating model on {len(X_test)} test samples...")
    
    # Make predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print results
    print("\n📊 EVALUATION RESULTS")
    print("=" * 30)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    
    # Create visualizations
    create_evaluation_plots(metrics, y_true, y_pred, y_pred_proba)
    
    # Generate report
    generate_evaluation_report(metrics, model_info)
    
    # Save metrics
    save_metrics_json(metrics)
    
    print(f"\n✅ Model evaluation completed!")
    print(f"📊 Performance metrics calculated")
    print(f"📈 Evaluation visualizations generated")
    print(f"📄 Comprehensive report created")

if __name__ == "__main__":
    main()
