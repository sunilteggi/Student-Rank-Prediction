import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Train model to predict rank
def train_model(historical_data):
    """Train a linear regression model to predict rank from historical data"""
    
    # Prepare features (based on available data columns) and target variable (rank)
    features = historical_data[['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5', 'average_score', 'total_score']]
    target = historical_data['rank']  # Assuming 'rank' is available in historical data
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    predictions = model.predict(X_test)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return model
