import numpy as np
import pandas as pd
import os

def generate_student_data(n_samples=500, random_state=42):
    np.random.seed(random_state)
    
    # 1. Study Hours (1.0 to 10.0 hours per day)
    study_hours = np.round(np.random.normal(loc=5.5, scale=2.0, size=n_samples), 1)
    study_hours = np.clip(study_hours, 1.0, 10.0)
    
    # 2. Attendance Percentage (45.0% to 100.0%)
    attendance = np.round(np.random.normal(loc=78.0, scale=12.0, size=n_samples), 1)
    attendance = np.clip(attendance, 45.0, 100.0)
    
    # 3. Previous Semester Marks (35.0 to 98.0)
    previous_marks = np.round(np.random.normal(loc=68.0, scale=14.0, size=n_samples), 1)
    previous_marks = np.clip(previous_marks, 35.0, 98.0)
    
    # 4. Assignment Score (40.0 to 100.0) correlated with study hours
    assignment_score = np.round(0.4 * previous_marks + 4 * study_hours + np.random.normal(0, 8, n_samples), 1)
    assignment_score = np.clip(assignment_score, 40.0, 100.0)
    
    # 5. Internal Assessment Marks (10.0 to 30.0 out of 30)
    internal_marks = np.round(0.2 * previous_marks + 0.1 * attendance + np.random.normal(0, 2.5, n_samples), 1)
    internal_marks = np.clip(internal_marks, 10.0, 30.0)
    
    # 6. Backlogs (0 to 5) - higher chance if previous marks < 55
    backlogs = []
    for pm in previous_marks:
        if pm < 50:
            b = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        elif pm < 65:
            b = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        else:
            b = np.random.choice([0, 1], p=[0.92, 0.08])
        backlogs.append(b)
    backlogs = np.array(backlogs)
    
    # 7. Extracurricular Activities ('Yes' or 'No')
    extracurricular = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.45, 0.55])
    
    # 8. Sleep Hours (4.0 to 9.5 hours)
    sleep_hours = np.round(np.random.normal(loc=6.8, scale=1.1, size=n_samples), 1)
    sleep_hours = np.clip(sleep_hours, 4.0, 9.5)
    
    # 9. Internet Usage Hours (1.0 to 6.0 hours)
    internet_hours = np.round(np.random.normal(loc=3.2, scale=1.2, size=n_samples), 1)
    internet_hours = np.clip(internet_hours, 1.0, 6.0)
    
    # Calculate composite score for realistic classification target
    # Weights reflecting importance of factors
    score = (
        0.35 * previous_marks +
        0.25 * attendance +
        0.20 * (assignment_score) +
        0.15 * (internal_marks * 3.33) +  # scale to 100
        0.10 * (study_hours * 10) -
        8.0 * backlogs +
        np.random.normal(0, 4.0, n_samples)
    )
    
    # Classification target definition based on composite score percentiles
    performance_category = []
    for sc in score:
        if sc >= 72.0:
            performance_category.append('Good')
        elif sc >= 54.0:
            performance_category.append('Average')
        else:
            performance_category.append('Poor')
            
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance_Percentage': attendance,
        'Previous_Semester_Marks': previous_marks,
        'Assignment_Score': assignment_score,
        'Internal_Assessment_Marks': internal_marks,
        'Backlogs': backlogs,
        'Extracurricular_Activities': extracurricular,
        'Sleep_Hours': sleep_hours,
        'Internet_Usage_Hours': internet_hours,
        'Performance_Category': performance_category
    })
    
    # Add a tiny amount of missing values (5 rows) to demonstrate data preprocessing & cleaning
    nan_indices = np.random.choice(n_samples, size=5, replace=False)
    df.loc[nan_indices[0], 'Study_Hours'] = np.nan
    df.loc[nan_indices[1], 'Attendance_Percentage'] = np.nan
    df.loc[nan_indices[2], 'Previous_Semester_Marks'] = np.nan
    df.loc[nan_indices[3], 'Assignment_Score'] = np.nan
    df.loc[nan_indices[4], 'Extracurricular_Activities'] = np.nan

    return df

if __name__ == '__main__':
    df = generate_student_data(n_samples=500)
    os.makedirs('dataset', exist_ok=True)
    csv_path = os.path.join('dataset', 'student_performance.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated successfully at {csv_path} with shape {df.shape}")
