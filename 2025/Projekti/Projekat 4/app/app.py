from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os
from flask_cors import CORS
import numpy as np
import traceback
from pathlib import Path
import random

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

project_root = Path(__file__).parent.parent
models_dir = project_root / 'model' / 'models'

models = {}
scalers = {}

def load_models():
    try:
        models['random_forest'] = joblib.load(models_dir / 'random_forest_model.pkl')
        models['xgboost'] = joblib.load(models_dir / 'xgboost_model.pkl')
        models['logistic_regression'] = joblib.load(models_dir / 'logistic_regression_model.pkl')

        scalers['amount'] = joblib.load(models_dir / 'scaler_amount.pkl')
        scalers['time'] = joblib.load(models_dir / 'scaler_time.pkl')

        print(f"Models and scalers loaded successfully from {models_dir}")
        return True
    except Exception as e:
        print(f"Error loading models from {models_dir}: {str(e)}")
        traceback.print_exc()
        return False

if not load_models():
    print("Failed to load models. Please check model files and versions.")

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/model_report/<model_name>')
def serve_model_report(model_name):
    valid_models = {
        'random_forest': 'random_forest_report.txt',
        'xgboost': 'xgboost_report.txt',
        'logistic_regression': 'logistic_regression_report.txt'
    }

    if model_name not in valid_models:
        return jsonify({'success': False, 'error': 'Invalid model name'}), 404

    try:
        with open(models_dir / valid_models[model_name], 'r') as f:
            report_content = f.read()
        return report_content, 200, {'Content-Type': 'text/plain'}
    except FileNotFoundError:
        return jsonify({'success': False, 'error': 'Report file not found'}), 404

@app.route('/models/<path:filename>')
def serve_model_files(filename):
    return send_from_directory(models_dir, filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400

        data = request.json

        # Get raw values
        amount = float(data.get('amount', np.random.uniform(0.01, 1000.0)))
        time = float(data.get('time', np.random.randint(0, 172800)))

        # Scale the features
        try:
            amount_scaled = scalers['amount'].transform([[amount]])[0][0]
            time_scaled = scalers['time'].transform([[time]])[0][0]
        except Exception as e:
            return jsonify({'success': False, 'error': f'Feature scaling failed: {str(e)}'}), 500

        # Create dictionary with all features in correct order (excluding Class)
        transaction_data = {}
        
        # First add V1-V28 features
        for i in range(1, 29):
            transaction_data[f'V{i}'] = float(data.get(f'v{i}', np.random.normal()))
        
        # Then add the scaled features
        transaction_data['Amount_scaled'] = amount_scaled
        transaction_data['Time_scaled'] = time_scaled

        # Create DataFrame ensuring correct column order
        columns_order = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
        df = pd.DataFrame([transaction_data])[columns_order]

        model_type = data.get('model_type', 'random_forest')
        if model_type not in models:
            return jsonify({'success': False, 'error': f'Invalid model type: {model_type}'}), 400

        model = models[model_type]
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1][0]

        return jsonify({
            'success': True,
            'is_fraud': bool(prediction[0]),
            'probability': float(probability),
            'model_type': model_type,
            'raw_amount': amount,
            'raw_time': time
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)