from app import app

def test_routes():
    client = app.test_client()
    routes_to_test = [
        ('/', 200),
        ('/dataset', 200),
        ('/preprocessing', 200),
        ('/eda', 200),
        ('/training', 200),
        ('/evaluation', 200),
        ('/prediction', 200),
        ('/history', 200),
        ('/about', 200),
        ('/dataset-info', 200),
        ('/eda-data', 200),
        ('/evaluation', 200),
        ('/prediction-history', 200)
    ]
    
    print("Testing Flask Web Routes & REST Endpoints:")
    all_passed = True
    for route, expected_code in routes_to_test:
        response = client.get(route)
        status = response.status_code
        if status == expected_code:
            print(f"   [PASS] {route} -> HTTP {status}")
        else:
            print(f"   [FAIL] {route} -> Expected {expected_code}, got {status}")
            all_passed = False
            
    # Test POST /predict
    test_input = {
        'Study_Hours': 8.0,
        'Attendance_Percentage': 90.0,
        'Previous_Semester_Marks': 85.0,
        'Assignment_Score': 88.0,
        'Internal_Assessment_Marks': 26.0,
        'Backlogs': 0,
        'Extracurricular_Activities': 'Yes',
        'Sleep_Hours': 7.5,
        'Internet_Usage_Hours': 2.0,
        'selected_model': 'Random Forest'
    }
    pred_res = client.post('/predict', json=test_input)
    if pred_res.status_code == 200:
        res_json = pred_res.get_json()
        print(f"   [PASS] POST /predict -> HTTP 200 | Category: {res_json['predicted_category']} ({res_json['confidence_percentage']}%)")
    else:
        print(f"   [FAIL] POST /predict -> HTTP {pred_res.status_code}")
        all_passed = False
        
    if all_passed:
        print("\n--- ALL FLASK ROUTES PASSED SUCCESSFULLY! ---")
    else:
        raise Exception("Route tests failed!")

if __name__ == '__main__':
    test_routes()
