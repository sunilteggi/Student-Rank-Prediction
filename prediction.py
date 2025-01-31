from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model(historical_data):
    """Train the prediction model using historical data"""
    # Features used for training (make sure they match prediction)
    features = historical_data[['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5', 'average_score']]
    target = historical_data['rank']  # Assuming 'rank' is the target variable

    model = LinearRegression()
    model.fit(features, target)  # Train the model
    return model

def predict_rank(model, current_data):
    """Predict the rank for a given user using their quiz data"""
    # Ensure that the current data contains the required features for prediction
    if 'quiz1' in current_data.columns and 'quiz2' in current_data.columns and \
       'quiz3' in current_data.columns and 'quiz4' in current_data.columns and \
       'quiz5' in current_data.columns and 'accuracy' in current_data.columns and 'average_score' in current_data.columns:
        features = current_data[['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5', 'average_score']]
    else:
        raise ValueError("Missing necessary columns in current data")

    predicted_rank = model.predict(features)
    return predicted_rank[0]  # Returning the predicted rank for the first student (or specific student)








