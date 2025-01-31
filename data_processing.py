import pandas as pd

def load_data():
    """Load current and historical quiz data"""
    current_data = pd.read_json('data/current_quiz_data.json')  # Load the current quiz data
    historical_data = pd.read_json('data/historical_quiz_data.json')  # Load the historical data
    return current_data, historical_data

def preprocess_data(current_data, historical_data):
    """Preprocess current and historical data"""
    
    # Debug: Print columns before processing
    print("Before Preprocessing Current Data Columns:", current_data.columns)
    
    # Preprocess current_data: Add 'accuracy' if 'correct_answers' and 'total_questions' exist
    if 'correct_answers' in current_data.columns and 'total_questions' in current_data.columns:
        current_data['accuracy'] = current_data['correct_answers'] / current_data['total_questions']
    else:
        print("Missing required columns in current quiz data")
    
    # Add 'average_score' for the current data
    current_data['average_score'] = current_data[['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5']].mean(axis=1)

    # Process historical data for training: Add 'average_score' feature
    historical_data = historical_data.dropna()  # Remove rows with NaN values
    historical_data['average_score'] = historical_data[['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5']].mean(axis=1)

    # Debug: Print columns after processing
    print("After Preprocessing Current Data Columns:", current_data.columns)
    print("Historical Data Columns:", historical_data.columns)
    
    return current_data, historical_data


