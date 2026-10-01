import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from preprocessing.preprocess import (
    StudentDataPreprocessor, split_dataset, FEATURE_COLUMNS, NUMERICAL_FEATURES, TARGET_COLUMN
)
from ml.train import train_all_models, load_trained_model, MODELS_DIR
from ml.evaluate import evaluate_models
from ml.predict import predict_student_performance
from database.db import init_db, save_prediction, get_all_predictions, clear_predictions

app = Flask(__name__)
app.secret_key = 'student_performance_ml_secret_key_antigravity'
app.config['UPLOAD_FOLDER'] = os.path.join('dataset', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max limit
ALLOWED_EXTENSIONS = {'csv'}

# Global State Container for session/active dataset & trained pipeline
STATE = {
    'dataset_path': os.path.join('dataset', 'student_performance.csv'),
    'preprocessor': None,
    'trained_models': {},
    'evaluation_metrics': {},
    'feature_importances': {},
    'selected_model_name': 'Random Forest',
    'is_trained': False
}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_dataset_and_models():
    """
    Auto-initializes standard dataset & trains models if not yet loaded into state.
    """
    if not os.path.exists(STATE['dataset_path']):
        from dataset.generate_dataset import generate_student_data
        df = generate_student_data()
        df.to_csv(STATE['dataset_path'], index=False)

    if not STATE['is_trained']:
        try:
            df = pd.read_csv(STATE['dataset_path'])
            preprocessor = StudentDataPreprocessor()
            X_scaled, y, summary, df_clean = preprocessor.fit_transform(df)
            X_train, X_test, y_train, y_test = split_dataset(X_scaled, y, random_state=42)
            
            trained_models = train_all_models(X_train, y_train, random_state=42)
            metrics, importances = evaluate_models(trained_models, X_test, y_test)
            
            STATE['preprocessor'] = preprocessor
            STATE['trained_models'] = trained_models
            STATE['evaluation_metrics'] = metrics
            STATE['feature_importances'] = importances
            STATE['is_trained'] = True
            
            # Save preprocessor
            preprocessor.save(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
        except Exception as e:
            print(f"Auto-training initialization error: {e}")

# Run startup initialization
ensure_dataset_and_models()

# --- WEB PAGE ROUTES ---

@app.route('/')
def home():
    ensure_dataset_and_models()
    history = get_all_predictions()
    total_predictions = len(history)
    best_model_name = 'Random Forest'
    best_accuracy = 0.0
    
    if STATE['evaluation_metrics']:
        best_model_name = max(STATE['evaluation_metrics'], key=lambda k: STATE['evaluation_metrics'][k]['accuracy'])
        best_accuracy = STATE['evaluation_metrics'][best_model_name]['accuracy']
        
    return render_template(
        'index.html',
        total_predictions=total_predictions,
        best_model_name=best_model_name,
        best_accuracy=best_accuracy,
        selected_model=STATE['selected_model_name']
    )

@app.route('/dataset')
def dataset_page():
    return render_template('dataset.html')

@app.route('/preprocessing')
def preprocessing_page():
    return render_template('preprocessing.html')

@app.route('/eda')
def eda_page():
    return render_template('eda.html')

@app.route('/training')
def training_page():
    return render_template('training.html')

@app.route('/evaluation')
def evaluation_page():
    return render_template('evaluation.html')

@app.route('/prediction')
def prediction_page():
    available_models = list(STATE['trained_models'].keys()) if STATE['trained_models'] else [
        'Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors'
    ]
    return render_template(
        'prediction.html',
        available_models=available_models,
        selected_model=STATE['selected_model_name']
    )

@app.route('/history')
def history_page():
    predictions = get_all_predictions()
    return render_template('history.html', predictions=predictions)

@app.route('/about')
def about_page():
    return render_template('about.html')


# --- REST API ENDPOINTS ---

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'dataset_file' not in request.files:
        flash('No file part provided', 'danger')
        return redirect(url_for('dataset_page'))
        
    file = request.files['dataset_file']
    if file.filename == '':
        flash('No selected file', 'warning')
        return redirect(url_for('dataset_page'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Validate CSV structure
        try:
            df = pd.read_csv(save_path)
            missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
            if missing_cols and TARGET_COLUMN not in df.columns:
                flash(f"CSV missing required columns: {missing_cols}. Please check your dataset.", "danger")
                return redirect(url_for('dataset_page'))
                
            STATE['dataset_path'] = save_path
            STATE['is_trained'] = False  # Reset trained state to require re-training
            flash(f"Dataset '{filename}' uploaded successfully! Clean dataset has {df.shape[0]} rows and {df.shape[1]} columns.", "success")
        except Exception as e:
            flash(f"Error reading CSV file: {str(e)}", "danger")
            
        return redirect(url_for('dataset_page'))
    else:
        flash("Invalid file format. Please upload a valid .csv file.", "danger")
        return redirect(url_for('dataset_page'))

@app.route('/dataset-info', methods=['GET'])
def get_dataset_info():
    try:
        df = pd.read_csv(STATE['dataset_path'])
        preview_data = df.head(10).to_dict(orient='records')
        
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        null_counts = {col: int(null_cnt) for col, null_cnt in df.isnull().sum().items()}
        
        # Numeric stats summary
        numeric_df = df.select_dtypes(include=[np.number])
        stats_dict = numeric_df.describe().round(2).to_dict() if not numeric_df.empty else {}
        
        return jsonify({
            'success': True,
            'file_name': os.path.basename(STATE['dataset_path']),
            'shape': list(df.shape),
            'columns': list(df.columns),
            'data_types': dtypes_dict,
            'missing_values': null_counts,
            'total_missing': int(df.isnull().sum().sum()),
            'duplicates_count': int(df.duplicated().sum()),
            'preview': preview_data,
            'summary_statistics': stats_dict
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def run_preprocess():
    try:
        df = pd.read_csv(STATE['dataset_path'])
        preprocessor = StudentDataPreprocessor()
        X_scaled, y, summary, df_clean = preprocessor.fit_transform(df)
        STATE['preprocessor'] = preprocessor
        
        return jsonify({
            'success': True,
            'summary': summary,
            'clean_head': df_clean.head(5).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/eda-data', methods=['GET'])
def get_eda_data():
    try:
        df = pd.read_csv(STATE['dataset_path'])
        
        # Target Distribution
        target_counts = df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df.columns else {}
        
        # Attendance vs Marks sample scatter
        attendance_marks = df[['Attendance_Percentage', 'Previous_Semester_Marks', TARGET_COLUMN]].dropna().to_dict(orient='records') if 'Attendance_Percentage' in df.columns else []
        
        # Study Hours vs Marks
        study_marks = df[['Study_Hours', 'Previous_Semester_Marks', TARGET_COLUMN]].dropna().to_dict(orient='records') if 'Study_Hours' in df.columns else []
        
        # Correlation Matrix for numerical columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().round(2).to_dict() if not numeric_df.empty else {}
        
        # Feature distributions (histogram bins)
        feature_distributions = {}
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                counts, bin_edges = np.histogram(df[col].dropna(), bins=8)
                feature_distributions[col] = {
                    'bins': [round(b, 1) for b in bin_edges[:-1]],
                    'counts': [int(c) for c in counts]
                }
                
        return jsonify({
            'success': True,
            'shape': list(df.shape),
            'target_distribution': target_counts,
            'attendance_marks': attendance_marks,
            'study_marks': study_marks,
            'correlation_matrix': corr_matrix,
            'feature_distributions': feature_distributions
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def trigger_training():
    try:
        df = pd.read_csv(STATE['dataset_path'])
        preprocessor = StudentDataPreprocessor()
        X_scaled, y, summary, df_clean = preprocessor.fit_transform(df)
        X_train, X_test, y_train, y_test = split_dataset(X_scaled, y, random_state=42)
        
        trained_models = train_all_models(X_train, y_train, random_state=42)
        metrics, importances = evaluate_models(trained_models, X_test, y_test)
        
        STATE['preprocessor'] = preprocessor
        STATE['trained_models'] = trained_models
        STATE['evaluation_metrics'] = metrics
        STATE['feature_importances'] = importances
        STATE['is_trained'] = True
        
        preprocessor.save(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
        
        return jsonify({
            'success': True,
            'message': 'All 4 ML models trained successfully!',
            'train_size': len(X_train),
            'test_size': len(X_test),
            'metrics': metrics,
            'feature_importances': importances
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/evaluation', methods=['GET'])
def get_evaluation_results():
    if not STATE['is_trained']:
        ensure_dataset_and_models()
        
    return jsonify({
        'success': True,
        'metrics': STATE['evaluation_metrics'],
        'feature_importances': STATE['feature_importances']
    })

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.json if request.is_json else request.form.to_dict()
        model_name = data.get('selected_model', STATE['selected_model_name'])
        
        if not STATE['is_trained']:
            ensure_dataset_and_models()
            
        model = STATE['trained_models'].get(model_name)
        if model is None:
            model = load_trained_model(model_name)
            
        preprocessor = STATE['preprocessor']
        if preprocessor is None:
            preprocessor = StudentDataPreprocessor.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
            
        # Execute prediction pipeline
        res = predict_student_performance(model, preprocessor, data)
        STATE['selected_model_name'] = model_name
        
        # Store prediction in SQLite
        db_id = save_prediction(data, model_name, res['predicted_category'], res['confidence_percentage'])
        res['db_record_id'] = db_id
        res['selected_model'] = model_name
        res['input_summary'] = data
        res['success'] = True
        
        return jsonify(res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/prediction-history', methods=['GET'])
def get_history_api():
    try:
        records = get_all_predictions()
        return jsonify({'success': True, 'records': records, 'count': len(records)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/prediction-history', methods=['DELETE'])
def clear_history_api():
    try:
        clear_predictions()
        return jsonify({'success': True, 'message': 'Prediction history cleared successfully.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
