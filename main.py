from data_processing import load_data, preprocess_data
from prediction import train_model, predict_rank

def main():
    current_data, historical_data = load_data()  # Load data
    
    # Preprocess data
    current_data, historical_data = preprocess_data(current_data, historical_data)
    
    # Train the model using historical data
    model = train_model(historical_data)
    
    # Predict the rank for the current data
    predicted_rank = predict_rank(model, current_data)
    
    print("Predicted Rank:", predicted_rank)

if __name__ == "__main__":
    main()




