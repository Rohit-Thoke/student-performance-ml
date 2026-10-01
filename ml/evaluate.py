import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from preprocessing.preprocess import TARGET_CLASSES

FEATURE_NAMES = [
    'Study Hours',
    'Attendance %',
    'Previous Marks',
    'Assignment Score',
    'Internal Marks',
    'Backlogs',
    'Sleep Hours',
    'Internet Hours',
    'Extracurricular'
]

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluates all trained models on unseen test dataset.
    Returns:
      - metrics_summary: Dict containing accuracy, precision, recall, f1, confusion matrix per model.
      - feature_importances: Dict containing feature importance scores for tree models.
    """
    metrics_summary = {}
    feature_importances = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
        
        metrics_summary[name] = {
            'accuracy': round(acc * 100, 2),
            'precision_macro': round(prec_macro * 100, 2),
            'precision_weighted': round(prec_weighted * 100, 2),
            'recall_macro': round(rec_macro * 100, 2),
            'recall_weighted': round(rec_weighted * 100, 2),
            'f1_macro': round(f1_macro * 100, 2),
            'f1_weighted': round(f1_weighted * 100, 2),
            'confusion_matrix': cm,
            'labels': TARGET_CLASSES
        }
        
        # Calculate feature importances if tree-based
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances[name] = [
                {'feature': feat, 'importance': round(float(imp), 4)}
                for feat, imp in zip(FEATURE_NAMES, importances)
            ]
            feature_importances[name].sort(key=lambda x: x['importance'], reverse=True)
            
    return metrics_summary, feature_importances
