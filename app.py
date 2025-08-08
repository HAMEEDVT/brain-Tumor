import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from datetime import datetime
import json
import tensorflow as tf
import time

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
MODEL_PATH = 'model/brain_tumor_model.h5'
IMG_SIZE = 128  # Updated to match training
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'brain_tumor_detection_secret_key_2024'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
CATEGORIES = ['No Tumor', 'Tumor Present']
model_loaded = False

# Simple metrics tracking
prediction_count = 0

def update_prediction_count():
    """Update prediction count."""
    global prediction_count
    prediction_count += 1

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_safely():
    """Load the trained model with error handling."""
    global model, model_loaded
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            model_loaded = True
            print(f"Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image_path):
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image_path (str): Path to the uploaded image
        
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    """
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image")
        
        # Resize to model input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values
        img = img / 255.0
        
        # Reshape for model input
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_tumor(image_path):
    """
    Predict tumor presence in the uploaded image.
    
    Args:
        image_path (str): Path to the uploaded image
        
    Returns:
        dict: Prediction results with confidence scores
    """
    if not model_loaded:
        return {'error': 'Model not loaded'}
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return {'error': 'Failed to process image'}
    
    try:
        # Start timing
        start_time = time.time()
        
        # Make prediction
        prediction = model.predict(processed_img, verbose=0)
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        
        # Get predicted class and confidence
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        # Get confidence for both classes
        no_tumor_conf = float(prediction[0][0])
        tumor_conf = float(prediction[0][1])
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': CATEGORIES[predicted_class],
            'confidence': confidence,
            'no_tumor_confidence': no_tumor_conf,
            'tumor_confidence': tumor_conf,
            'prediction_probs': prediction[0].tolist(),
            'prediction_time': round(prediction_time, 3)
        }
        
        # Update prediction count
        update_prediction_count()
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {'error': f'Prediction failed: {str(e)}'}

def create_confidence_chart(no_tumor_conf, tumor_conf):
    """
    Create a confidence chart visualization.
    
    Args:
        no_tumor_conf (float): Confidence for no tumor
        tumor_conf (float): Confidence for tumor present
        
    Returns:
        str: Base64 encoded chart image
    """
    try:
        # Create the chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['No Tumor', 'Tumor Present']
        confidences = [no_tumor_conf, tumor_conf]
        colors = ['#28a745', '#dc3545']
        
        bars = ax.bar(categories, confidences, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Confidence Score')
        ax.set_title('Prediction Confidence')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Save to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
        
    except Exception as e:
        print(f"Error creating confidence chart: {str(e)}")
        return None

def get_model_info():
    """Get information about the loaded model."""
    if not model_loaded:
        return None
    
    try:
        model_info = {
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': int(model.count_params()),
            'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
            'non_trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
        }
        return model_info
    except Exception as e:
        print(f"Error getting model info: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page with upload form."""
    model_info = get_model_info()
    return render_template('index.html', model_loaded=model_loaded, model_info=model_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction."""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"mri_{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_tumor(filepath)
            
            if 'error' in result:
                flash(f'Prediction error: {result["error"]}', 'error')
                return redirect(request.url)
            
            # Create confidence chart
            chart_img = create_confidence_chart(
                result['no_tumor_confidence'], 
                result['tumor_confidence']
            )
            
            return render_template('result.html',
                               result=result,
                               image_file=filename,
                               chart_img=chart_img,
                               timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)
    else:
        flash('Invalid file type. Please upload an image file.', 'error')
        return redirect(request.url)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file temporarily
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"api_{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_tumor(filepath)
            
            # Clean up temporary file
            try:
                os.remove(filepath)
            except:
                pass
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/model-info')
def model_info():
    """Display model information."""
    if not model_loaded:
        flash('Model not loaded', 'error')
        return redirect(url_for('index'))
    
    info = get_model_info()
    return render_template('model_info.html', model_info=info)



@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

def main():
    """Initialize and run the Flask application."""
    global model_loaded
    
    print("Starting Brain Tumor Detection Web Application...")
    
    # Load model
    if load_model_safely():
        print("✅ Model loaded successfully")
    else:
        print("❌ Failed to load model. Please ensure the model file exists.")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
