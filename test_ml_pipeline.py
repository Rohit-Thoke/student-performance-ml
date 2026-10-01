import pandas as pd
from preprocessing.preprocess import StudentDataPreprocessor, split_dataset
from ml.train import train_all_models
from ml.evaluate import evaluate_models
from ml.predict import predict_student_performance
from database.db import init_db, save_prediction, get_all_predictions

def test_pipeline():
    print("1. Loading dataset...")
    df = pd.read_csv('dataset/student_performance.csv')
    print(f"   Dataset shape: {df.shape}")
    
    print("\n2. Testing Preprocessor...")
    preprocessor = StudentDataPreprocessor()
    X_scaled, y, summary, df_clean = preprocessor.fit_transform(df)
    print(f"   Processed X shape: {X_scaled.shape}, y shape: {y.shape}")
    print(f"   Missing values handled: {summary['missing_count_handled']}")
    
    print("\n3. Testing Train/Test Split...")
    X_train, X_test, y_train, y_test = split_dataset(X_scaled, y, test_size=0.2, random_state=42)
    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    print("\n4. Training 4 ML Models...")
    trained_models = train_all_models(X_train, y_train, random_state=42)
    for name in trained_models.keys():
        print(f"   - {name} trained successfully.")
        
    print("\n5. Evaluating Models on Test Dataset...")
    metrics, feature_importances = evaluate_models(trained_models, X_test, y_test)
    for name, m in metrics.items():
        print(f"   - {name}: Accuracy = {m['accuracy']}%, F1 = {m['f1_weighted']}%")
        
    print("\n6. Testing Prediction Engine...")
    test_input = {
        'Study_Hours': 7.5,
        'Attendance_Percentage': 92.0,
        'Previous_Semester_Marks': 84.0,
        'Assignment_Score': 88.0,
        'Internal_Assessment_Marks': 26.0,
        'Backlogs': 0,
        'Extracurricular_Activities': 'Yes',
        'Sleep_Hours': 7.5,
        'Internet_Usage_Hours': 2.5
    }
    rf_model = trained_models['Random Forest']
    pred_res = predict_student_performance(rf_model, preprocessor, test_input)
    print(f"   Prediction result: {pred_res['predicted_category']} (Confidence: {pred_res['confidence_percentage']}%)")
    
    print("\n7. Testing SQLite Database...")
    init_db()
    db_id = save_prediction(test_input, 'Random Forest', pred_res['predicted_category'], pred_res['confidence_percentage'])
    history = get_all_predictions()
    print(f"   Inserted record ID: {db_id}. Total history records: {len(history)}")
    
    print("\n--- ALL PIPELINE TESTS PASSED SUCCESSFULLY! ---")

if __name__ == '__main__':
    test_pipeline()
