# Email Spam Detection

## Overview
The **Email Spam Detection** application is a Streamlit-based tool that helps users detect spam emails using machine learning. The application allows users to load a dataset, split the data, train a model, and test its performance interactively.

## Features
- **Load Data**: Load an email dataset for spam classification.
- **Split Data**: Prepares the dataset for training and testing.
- **Train Model**: Trains a machine learning model on the dataset.
- **Test Model**: Evaluates the model's performance using test data.
- **User-friendly UI**: Interactive widgets for easy navigation through different stages.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/email-spam-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd email-spam-detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Use the sidebar navigation to load data, split it, train a model, and test predictions.
3. Follow the app's instructions to ensure a smooth workflow.

## Dependencies
- Streamlit
- Pandas
- Scikit-learn
- Python Standard Libraries

## How It Works
1. **Data Loading**: Loads a dataset containing emails labeled as spam or not spam.
2. **Data Splitting**: Splits the dataset into training and testing sets.
3. **Training**: Uses a machine learning model to learn patterns from training data.
4. **Testing**: Evaluates the model's accuracy and effectiveness on unseen emails.

