
# AI Emotional Wellness Ally

## Overview

The **AI Emotional Wellness Ally** is a personalized AI-powered companion designed to help individuals improve their emotional well-being. This system leverages behavioral data, machine learning, and reinforcement learning to predict emotional states and provide actionable, personalized recommendations.

## Features

- **Behavioral Data Tracking**: Non-intrusive data collection of screen time, physical activity, sleep patterns, and heart rate.
- **Mood Prediction**: Uses a Random Forest Classifier to predict emotional states such as "anxious," "calm," "stressed," or "happy."
- **Personalized Recommendations**: Suggests activities and lifestyle changes to improve emotional well-being.
- **Reinforcement Learning**: Adapts recommendations over time based on user feedback.

## Problem Statement

Modern lifestyles often lead to emotional challenges like stress, anxiety, and depression, exacerbated by limited access to timely and personalized mental health support. The **AI Emotional Wellness Ally** addresses this gap by providing a smart, accessible, and tailored solution for emotional well-being.

## Goals

1. **Track Behavioral Data**: Collect non-intrusive data such as device usage, physical activity, and sleep.
2. **Predict Emotional States**: Leverage machine learning to analyze user data and predict moods.
3. **Provide Personalized Recommendations**: Suggest actions to improve emotional well-being.
4. **Incorporate Reinforcement Learning**: Continuously refine recommendations based on user feedback.

## Methodology

### 1. Mood Prediction
- **Model**: Random Forest Classifier.
- **Data**: Synthetic behavioral metrics including screen time, steps taken, sleep patterns, and heart rate.
- **Accuracy**: Achieved moderate success in predicting emotional states.

### 2. Personalized Recommendations
- **Initial System**: Rule-based suggestions tailored to predicted emotional states.
- **Learning Agent**: Q-learning algorithm for refining suggestions based on feedback.

### 3. Feedback Simulation
- Conducted simulations to train the reinforcement learning agent, improving the system's adaptability.

## Timeline

| Date         | Task                                   | Progress                                                                 |
|--------------|---------------------------------------|-------------------------------------------------------------------------|
| Sept 15, 2024 | Project Kickoff                       | Defined scope, goals, and development environment setup.               |
| Sept 18, 2024 | Data Collection & Preprocessing       | Collected synthetic data and preprocessed for training.                |
| Sept 25, 2024 | Model Development                    | Trained Random Forest Classifier for mood prediction.                  |
| Oct 10, 2024 | RL Agent Development                 | Implemented Q-learning for personalized recommendations.               |
| Nov 1, 2024  | System Refinement and Testing         | Conducted final testing of integrated components.                      |
| Nov 8, 2024  | Demo and Report Submission            | Delivered prototype and project documentation.                         |

## Future Enhancements

- **Real-time Data Integration**: Incorporate data from wearables (e.g., Fitbit, Apple Watch).
- **Expanded Emotional Spectrum**: Support more nuanced emotional states.
- **Social Integration**: Add features to promote social connections and combat loneliness.

## Conclusion

The **AI Emotional Wellness Ally** demonstrates the potential of AI in supporting mental health by combining mood prediction and reinforcement learning for personalized recommendations. Future iterations will focus on integrating real-world data and expanding functionality to enhance effectiveness.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```


For any queries or contributions, feel free to reach out to **Kaushiki Milind Joshi**.
