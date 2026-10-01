import sqlite3
import os

DB_DIR = 'database'
DB_PATH = os.path.join(DB_DIR, 'predictions.db')

def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Creates the predictions table if it doesn't already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            study_hours REAL,
            attendance REAL,
            previous_marks REAL,
            assignment_score REAL,
            internal_marks REAL,
            backlogs INTEGER,
            extracurricular TEXT,
            sleep_hours REAL,
            internet_hours REAL,
            selected_model TEXT,
            predicted_class TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(input_dict, selected_model, predicted_class, confidence):
    """
    Inserts a new prediction record into SQLite database.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            study_hours, attendance, previous_marks, assignment_score,
            internal_marks, backlogs, extracurricular, sleep_hours,
            internet_hours, selected_model, predicted_class, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        float(input_dict.get('Study_Hours', 0)),
        float(input_dict.get('Attendance_Percentage', 0)),
        float(input_dict.get('Previous_Semester_Marks', 0)),
        float(input_dict.get('Assignment_Score', 0)),
        float(input_dict.get('Internal_Assessment_Marks', 0)),
        int(input_dict.get('Backlogs', 0)),
        str(input_dict.get('Extracurricular_Activities', 'No')),
        float(input_dict.get('Sleep_Hours', 0)),
        float(input_dict.get('Internet_Usage_Hours', 0)),
        str(selected_model),
        str(predicted_class),
        float(confidence)
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_all_predictions():
    """
    Retrieves all logged predictions sorted by newest first.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def clear_predictions():
    """
    Clears all records from the predictions table.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    return True
