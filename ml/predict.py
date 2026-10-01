import numpy as np
from preprocessing.preprocess import TARGET_REVERSE_MAPPING

def predict_student_performance(model, preprocessor, input_dict):
    """
    Passes preprocessed single instance to selected model and computes:
    1. Predicted class label ('Poor', 'Average', 'Good')
    2. Prediction confidence / probability distribution
    3. Meaningful input factor insights explaining the outcome
    """
    # Transform input
    X_scaled = preprocessor.transform_single(input_dict)
    
    # Model prediction
    pred_code = int(model.predict(X_scaled)[0])
    predicted_label = TARGET_REVERSE_MAPPING.get(pred_code, 'Unknown')
    
    # Probabilities / Confidence score
    probabilities = {}
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_scaled)[0]
        for idx, prob in enumerate(probs):
            cls_name = TARGET_REVERSE_MAPPING.get(idx, f"Class {idx}")
            probabilities[cls_name] = round(float(prob) * 100, 2)
        confidence = probabilities.get(predicted_label, round(float(np.max(probs)) * 100, 2))
    else:
        # Fallback for models without predict_proba
        confidence = 100.0
        probabilities = {predicted_label: 100.0}

    # Generate data-driven feature influence insights
    insights = generate_feature_insights(input_dict, predicted_label)

    return {
        'predicted_category': predicted_label,
        'confidence_percentage': confidence,
        'probabilities': probabilities,
        'insights': insights
    }

def generate_feature_insights(input_dict, predicted_label):
    """
    Analyzes input values against standard academic benchmarks to explain why
    a student received a Good, Average, or Poor performance prediction.
    """
    insights = []
    
    study_hours = float(input_dict.get('Study_Hours', 5.0))
    attendance = float(input_dict.get('Attendance_Percentage', 75.0))
    prev_marks = float(input_dict.get('Previous_Semester_Marks', 65.0))
    assignment = float(input_dict.get('Assignment_Score', 70.0))
    internal = float(input_dict.get('Internal_Assessment_Marks', 20.0))
    backlogs = int(input_dict.get('Backlogs', 0))
    extracurricular = str(input_dict.get('Extracurricular_Activities', 'No')).strip().lower() in ['yes', '1', 'true']
    sleep = float(input_dict.get('Sleep_Hours', 7.0))
    internet = float(input_dict.get('Internet_Usage_Hours', 3.0))

    # Attendance Insights
    if attendance >= 85.0:
        insights.append({"type": "positive", "text": f"High attendance ({attendance}%) strongly supports academic engagement and consistent learning."})
    elif attendance < 70.0:
        insights.append({"type": "negative", "text": f"Low attendance ({attendance}%) is a major risk factor impacting continuous evaluation."})

    # Study Hours Insights
    if study_hours >= 6.0:
        insights.append({"type": "positive", "text": f"Dedicated daily study routine of {study_hours} hours strengthens concept retention."})
    elif study_hours < 3.0:
        insights.append({"type": "negative", "text": f"Limited daily study time ({study_hours} hrs) reduces preparation capacity for assessments."})

    # Previous Semester Marks & Backlogs
    if prev_marks >= 75.0:
        insights.append({"type": "positive", "text": f"Strong academic foundation demonstrated by previous semester score of {prev_marks}%."})
    elif prev_marks < 55.0:
        insights.append({"type": "negative", "text": f"Lower prior academic score ({prev_marks}%) indicates foundational concept gaps."})

    if backlogs > 0:
        insights.append({"type": "negative", "text": f"Active backlog count ({backlogs}) creates academic stress and divides study focus."})
    else:
        insights.append({"type": "positive", "text": "Clean academic record with zero active backlogs."})

    # Internal & Assignment Performance
    if assignment >= 80.0 and internal >= 22.0:
        insights.append({"type": "positive", "text": f"Consistently strong continuous evaluation scores (Assignment: {assignment}%, Internal: {internal}/30)."})

    # Sleep & Internet Usage Lifestyle Balance
    if sleep < 6.0:
        insights.append({"type": "warning", "text": f"Insufficient sleep ({sleep} hrs/night) may trigger cognitive fatigue and reduce exam focus."})
    elif sleep >= 7.0 and sleep <= 9.0:
        insights.append({"type": "positive", "text": f"Healthy sleep duration of {sleep} hours supports memory consolidation."})

    if internet > 5.0 and study_hours < 4.0:
        insights.append({"type": "warning", "text": f"High daily non-study internet usage ({internet} hrs) relative to study time ({study_hours} hrs)."})

    # Summary Conclusion
    if predicted_label == 'Good':
        summary = "Overall profile demonstrates strong academic consistency across attendance, prior marks, and study discipline."
    elif predicted_label == 'Average':
        summary = "Student shows satisfactory performance but can upgrade to Good by increasing attendance and regular study hours."
    else:
        summary = "Student needs immediate academic intervention, attendance recovery, and remedial study support."

    return {
        'summary': summary,
        'key_points': insights
    }
