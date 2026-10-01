import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

MODELS_DIR = os.path.join('models', 'trained_models')

def train_all_models(X_train, y_train, random_state=42):
    """
    Trains 4 classification algorithms:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    4. K-Nearest Neighbors (KNN)
    Returns dictionary of trained model objects.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=5,
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Save each model using joblib
        safe_filename = name.lower().replace(' ', '_') + '.joblib'
        save_path = os.path.join(MODELS_DIR, safe_filename)
        joblib.dump(model, save_path)
        
    return trained_models

def load_trained_model(model_name):
    safe_filename = model_name.lower().replace(' ', '_') + '.joblib'
    file_path = os.path.join(MODELS_DIR, safe_filename)
    if os.path.exists(file_path):
        return joblib.load(file_path)
    raise FileNotFoundError(f"Model binary not found at {file_path}")
