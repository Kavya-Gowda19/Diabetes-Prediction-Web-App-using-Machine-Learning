# Diabetes-Prediction-Web-App-using-Machine-Learning

This repository contains a Flask web application that predicts the likelihood of diabetes using various machine learning models. Users can register, log in, and input their health data to receive predictions from different models, including Logistic Regression, Random Forest, Decision Tree, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Gradient Boosting.

## Features

- **User Authentication**: Secure user registration and login system.
- **Data Preprocessing**: Handles missing values by replacing them with the mean or median of the column and standardizes the data.
- **Multiple Machine Learning Models**: Predictions using Logistic Regression, Random Forest, Decision Tree, SVM, KNN, and Gradient Boosting.
- **Model Performance Visualization**: Displays the accuracy of each model and provides a visual representation of model performance.
- **Prediction Results**: Provides prediction results with confidence percentages for each model.

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- `pip` (Python package installer)
- `virtualenv` (for creating virtual environments)

### Usage

1. **Register** for a new account.
2. **Log in** with your credentials.
3. **Navigate to the prediction page**.
4. **Input your health data** (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).
5. **Submit the data** to receive predictions from multiple models.
6. **View the prediction results** with confidence percentages.

## File Structure

- `app.py`: Main application script.
- `forms.py`: Contains form classes for user registration and login.
- `models.py`: Defines the database models (User model).
- `templates/`: HTML templates for rendering pages.
  - `layout.html`: Base template.
  - `login.html`: Login page template.
  - `register.html`: Registration page template.
  - `predict.html`: Prediction input page template.
  - `result.html`: Prediction result page template.
- `static/`: Static files (CSS, JS, images).
  - `styles.css`: Custom CSS styles.
- `requirements.txt`: List of Python packages required for the project.
- `migrations/`: Database migration files generated by Flask-Migrate.

## Dataset

The dataset used for training the models is the Pima Indians Diabetes Dataset. You can download it from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

### Data Preprocessing

- Missing values in the `Glucose`, `BloodPressure`, `BMI`, `SkinThickness`, and `Insulin` columns are replaced with the median or mean of the respective column.
- The features are standardized using `StandardScaler`.

## Machine Learning Models

The following machine learning models are used to predict diabetes:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Decision Tree Classifier**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)**
6. **Gradient Boosting Classifier**

### Model Training

The models are trained using the following steps:

1. **Splitting the dataset**: The data is split into training and testing sets (80% training, 20% testing).
2. **Standardizing the data**: Features are standardized using `StandardScaler`.
3. **Training the models**: Each model is trained on the training set.
4. **Evaluating the models**: The accuracy of each model is calculated on the test set.

### Model Accuracy

The application calculates and displays the accuracy of each model. A bar graph visualizes the accuracy of all models.

## Screenshots

### Home Page
![pic1](https://github.com/user-attachments/assets/3791e921-2508-42a7-8c68-30756d83a228)

### Registration Page
![pic3](https://github.com/user-attachments/assets/017710cd-4e38-480f-9019-b85c9bc836ae)

### Login Page
![pic2](https://github.com/user-attachments/assets/39651fed-c0b6-4173-9182-b31c811d59ff)

### Prediction Page
![pic7](https://github.com/user-attachments/assets/3bc2c5fd-358c-45dd-aa5e-87560a092de7)

### Result Page
![pic5](https://github.com/user-attachments/assets/e351580b-c34f-4982-b3fb-9ae0ac9a367c)
![pic6](https://github.com/user-attachments/assets/ebd53309-6c60-4442-bcbc-6f5aafb7c378)

## Acknowledgments

- Thanks to the [Flask](https://flask.palletsprojects.com/) community for their excellent web framework.
- Thanks to the [Scikit-Learn](https://scikit-learn.org/) community for their machine learning tools.
- Dataset sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

