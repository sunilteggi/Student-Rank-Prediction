import pandas as pd
import matplotlib.pyplot as plt
def visualize_rank_prediction(historical_data, model):
    """
    Visualize predicted rank against the actual rank.
    """
    # Predict ranks using the trained model
    features = historical_data[['quiz1', 'quiz2', 'quiz3', 'quiz4', 'quiz5', 'average_score']]
    predicted_ranks = model.predict(features)

    # Plot actual vs predicted ranks
    plt.figure(figsize=(10, 6))
    plt.scatter(historical_data['rank'], predicted_ranks, color='purple', alpha=0.6)
    plt.plot([historical_data['rank'].min(), historical_data['rank'].max()],
             [historical_data['rank'].min(), historical_data['rank'].max()], color='red', linestyle='--')
    plt.xlabel('Actual Rank')
    plt.ylabel('Predicted Rank')
    plt.title('Actual Rank vs Predicted Rank')
    plt.show()

