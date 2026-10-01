import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

FEATURE_COLUMNS = [
    'Study_Hours',
    'Attendance_Percentage',
    'Previous_Semester_Marks',
    'Assignment_Score',
    'Internal_Assessment_Marks',
    'Backlogs',
    'Extracurricular_Activities',
    'Sleep_Hours',
    'Internet_Usage_Hours'
]

NUMERICAL_FEATURES = [
    'Study_Hours',
    'Attendance_Percentage',
    'Previous_Semester_Marks',
    'Assignment_Score',
    'Internal_Assessment_Marks',
    'Backlogs',
    'Sleep_Hours',
    'Internet_Usage_Hours'
]

CATEGORICAL_FEATURES = ['Extracurricular_Activities']
TARGET_COLUMN = 'Performance_Category'
TARGET_CLASSES = ['Poor', 'Average', 'Good']
TARGET_MAPPING = {'Poor': 0, 'Average': 1, 'Good': 2}
TARGET_REVERSE_MAPPING = {0: 'Poor', 1: 'Average', 2: 'Good'}

class StudentDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputers_num = {}
        self.imputer_cat = 'No'
        self.is_fitted = False
        self.preprocessing_summary = {}

    def fit_transform(self, df):
        """
        Full preprocessing pipeline:
        1. Copy dataset & record initial info
        2. Handle missing values
        3. Remove duplicate records
        4. Validate / clip numerical ranges
        5. Encode binary categorical features
        6. Separate X and y
        7. Scale numerical features
        """
        df_clean = df.copy()
        
        initial_shape = df_clean.shape
        missing_count_before = int(df_clean.isnull().sum().sum())
        duplicates_removed = int(df_clean.duplicated().sum())
        
        # Drop duplicates
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        
        # Missing values imputation
        for col in NUMERICAL_FEATURES:
            if col in df_clean.columns:
                median_val = float(df_clean[col].median())
                self.imputers_num[col] = median_val
                df_clean[col] = df_clean[col].fillna(median_val)
                
        if 'Extracurricular_Activities' in df_clean.columns:
            mode_val = df_clean['Extracurricular_Activities'].mode()[0] if not df_clean['Extracurricular_Activities'].mode().empty else 'No'
            self.imputer_cat = mode_val
            df_clean['Extracurricular_Activities'] = df_clean['Extracurricular_Activities'].fillna(mode_val)
            
        # Range validation / Outlier cleaning
        df_clean['Study_Hours'] = np.clip(df_clean['Study_Hours'], 0, 16)
        df_clean['Attendance_Percentage'] = np.clip(df_clean['Attendance_Percentage'], 0, 100)
        df_clean['Previous_Semester_Marks'] = np.clip(df_clean['Previous_Semester_Marks'], 0, 100)
        df_clean['Assignment_Score'] = np.clip(df_clean['Assignment_Score'], 0, 100)
        df_clean['Internal_Assessment_Marks'] = np.clip(df_clean['Internal_Assessment_Marks'], 0, 30)
        df_clean['Backlogs'] = np.clip(df_clean['Backlogs'], 0, 10)
        df_clean['Sleep_Hours'] = np.clip(df_clean['Sleep_Hours'], 0, 16)
        df_clean['Internet_Usage_Hours'] = np.clip(df_clean['Internet_Usage_Hours'], 0, 16)
        
        # Encoding Extracurricular_Activities: 'Yes' -> 1, 'No' -> 0
        df_clean['Extracurricular_Activities_Enc'] = df_clean['Extracurricular_Activities'].apply(
            lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0
        )
        
        # Prepare Feature matrix X
        feature_order = NUMERICAL_FEATURES + ['Extracurricular_Activities_Enc']
        X = df_clean[feature_order].copy()
        
        # Fit & transform scaler on numerical features + encoded extracurricular
        X_scaled = self.scaler.fit_transform(X.values)
        self.is_fitted = True
        
        # Encode Target variable y
        y = None
        if TARGET_COLUMN in df_clean.columns:
            y = df_clean[TARGET_COLUMN].map(TARGET_MAPPING)
            # If any unmapped, fallback to mode
            if y.isnull().any():
                y = y.fillna(1).astype(int)
            else:
                y = y.astype(int)
                
        self.preprocessing_summary = {
            'initial_shape': list(initial_shape),
            'final_shape': list(df_clean.shape),
            'missing_count_handled': missing_count_before,
            'duplicates_removed': duplicates_removed,
            'features_processed': feature_order,
            'target_distribution': df_clean[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df_clean.columns else {}
        }
        
        return X_scaled, y, self.preprocessing_summary, df_clean

    def transform_single(self, input_dict):
        """
        Transforms a single user prediction input dictionary into scaled numpy array for model inference.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet!")
            
        study_hours = float(input_dict.get('Study_Hours', self.imputers_num.get('Study_Hours', 5.0)))
        attendance = float(input_dict.get('Attendance_Percentage', self.imputers_num.get('Attendance_Percentage', 75.0)))
        prev_marks = float(input_dict.get('Previous_Semester_Marks', self.imputers_num.get('Previous_Semester_Marks', 65.0)))
        assignment = float(input_dict.get('Assignment_Score', self.imputers_num.get('Assignment_Score', 70.0)))
        internal = float(input_dict.get('Internal_Assessment_Marks', self.imputers_num.get('Internal_Assessment_Marks', 20.0)))
        backlogs = float(input_dict.get('Backlogs', self.imputers_num.get('Backlogs', 0)))
        sleep = float(input_dict.get('Sleep_Hours', self.imputers_num.get('Sleep_Hours', 7.0)))
        internet = float(input_dict.get('Internet_Usage_Hours', self.imputers_num.get('Internet_Usage_Hours', 3.0)))
        
        extra_raw = str(input_dict.get('Extracurricular_Activities', 'No')).strip().lower()
        extra_enc = 1 if extra_raw in ['yes', '1', 'true'] else 0
        
        raw_values = [
            np.clip(study_hours, 0, 16),
            np.clip(attendance, 0, 100),
            np.clip(prev_marks, 0, 100),
            np.clip(assignment, 0, 100),
            np.clip(internal, 0, 30),
            np.clip(backlogs, 0, 10),
            np.clip(sleep, 0, 16),
            np.clip(internet, 0, 16),
            extra_enc
        ]
        
        X_single = np.array(raw_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X_single)
        return X_scaled

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path):
        return joblib.load(file_path)

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Stratified train-test split to ensure class proportion integrity.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
