import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random


# --- Step 1: Define Behavioral Data (Synthetic Data) ---


# Create synthetic dataset
data = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02', 
             '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
    'screen_time': [4.5, 3.0, 5.0, 6.5, 3.5, 2.0, 4.0, 5.5],
    'steps': [8000, 10000, 5000, 9000, 6000, 7000, 4500, 8000],
    'sleep_duration': [7.0, 8.0, 6.0, 7.5, 6.5, 7.0, 7.5, 6.5],
    'heart_rate': [72, 70, 75, 68, 71, 74, 73, 76],
    'mood': ['stressed', 'calm', 'anxious', 'calm', 'happy', 'calm', 
             'stressed', 'happy']
})


# Encode 'mood' as numeric values
data['mood'] = data['mood'].map({'stressed': 0, 'calm': 1, 'anxious': 2, 'happy': 3})


# --- Step 2: Mental State Prediction Model ---


# Features and target
X = data[['screen_time', 'steps', 'sleep_duration', 'heart_rate']]
y = data['mood']


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)


# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("Mental State Prediction Model Evaluation:")
print(pd.DataFrame({'True Mood': y_test.values, 'Predicted Mood': y_pred}))


# --- Step 3: Recommendation System ---


# Recommendation suggestions based on mood
recommendations = {
    0: ["Take a short walk", "Practice deep breathing", "Try mindfulness exercises"],
    1: ["Go for a jog", "Engage in a hobby", "Read a book"],
    2: ["Listen to calming music", "Do a relaxation exercise", "Take a short break"],
    3: ["Socialize with friends", "Engage in a creative activity", "Continue with your current activity"]
}


# --- Step 4: Reinforcement Learning for Personalized Recommendations ---


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros(len(actions))  # Initialize Q-table
    
    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore: Choose a random action
        else:
            return self.actions[np.argmax(self.q_table)]  # Exploit: Choose the best action


    def update_q_table(self, action, reward):
        action_idx = self.actions.index(action)
        self.q_table[action_idx] = (1 - self.alpha) * self.q_table[action_idx] + self.alpha * reward


# Instantiate the agent
actions = ["Take a short walk", "Listen to calming music", "Practice deep breathing", 
           "Go for a jog", "Socialize with friends"]
agent = QLearningAgent(actions)


# Simulate interaction: user feedback (use real feedback in a real-world scenario)
feedback = {
    "Take a short walk": 1,         # Very helpful
    "Listen to calming music": 0.8,  # Helpful
    "Practice deep breathing": 0.9,  # Very helpful
    "Go for a jog": 0.5,           # Neutral
    "Socialize with friends": 0.7  # Helpful
}


# Simulate 10 days of interaction with user feedback
print("\nStarting Reinforcement Learning for Personalized Recommendations...\n")
for day in range(10):
    # Choose an action (recommendation) based on mood and exploration-exploitation
    action = agent.choose_action()
    
    # Simulate the prediction of mood based on data (in real implementation, use model prediction)
    predicted_mood = random.choice([0, 1, 2, 3])  # Randomly simulate mood prediction for demo
    print(f"Day {day + 1}: Predicted Mood = {['stressed', 'calm', 'anxious', 'happy'][predicted_mood]}")
    
    # Get recommendation based on predicted mood, ensuring it's in the agent's actions
    recommended_action = random.choice([rec for rec in recommendations[predicted_mood] if rec in actions])
    print(f"   Suggested Activity: {recommended_action}")
    
    # Simulate user feedback (could be based on actual user input)
    reward = feedback.get(recommended_action, 0.5)  # Feedback rating (1 = very helpful, 0.5 = neutral, etc.)
    
    # Update Q-table with feedback
    agent.update_q_table(recommended_action, reward)
    
    print(f"   Feedback: {reward} (User feedback)\n")