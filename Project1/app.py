from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
from forms import LoginForm, RegistrationForm
from models import db, User  # Ensure this import is correct
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SECRET_KEY'] = '9938857ea158190730b2e6e58108c626'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the Flask app
db.init_app(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load user callback
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Importing the dataset
data = pd.read_csv(r"C:\Users\kavya\Downloads\diabetes[1].csv")

# Replacing 0 values with the mean or median of that column
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].median())
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].median())
data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())

# Separating features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Logistic Regression model and calculating accuracy
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
logistic_y_pred = logistic_model.predict(X_test_scaled)
logistic_accuracy = accuracy_score(y_test, logistic_y_pred) * 100

# Training a Random Forest model and calculating accuracy
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_scaled, y_train)
random_forest_y_pred = random_forest_model.predict(X_test_scaled)
random_forest_accuracy = accuracy_score(y_test, random_forest_y_pred) * 100

# Training a Decision Tree model and calculating accuracy
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train_scaled, y_train)
decision_tree_y_pred = decision_tree_model.predict(X_test_scaled)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_y_pred) * 100

# Training a SVM model and calculating accuracy
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_y_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_y_pred) * 100

# Training a KNN model and calculating accuracy
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
knn_y_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_y_pred) * 100

# Training a Gradient Boosting model and calculating accuracy
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X_train_scaled, y_train)
gradient_boosting_y_pred = gradient_boosting_model.predict(X_test_scaled)
gradient_boosting_accuracy = accuracy_score(y_test, gradient_boosting_y_pred) * 100

# Function to create an accuracy graph
def create_accuracy_graph():
    algorithms = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM', 'KNN', 'Gradient Boosting']
    accuracies = [
        logistic_accuracy, random_forest_accuracy, decision_tree_accuracy,
        svm_accuracy, knn_accuracy, gradient_boosting_accuracy
    ]

    fig, ax = plt.subplots()
    ax.barh(algorithms, accuracies, color='blue')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Model Accuracy')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64

accuracy_graph = create_accuracy_graph()


@app.route('/')
def layout():
    return render_template('layout.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('predict'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('predict'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('predict'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256', salt_length=16)
        user = User(username=form.username.data, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            input_features = [float(x) for x in request.form.values()]
            input_data = np.array([input_features])
            input_data_scaled = scaler.transform(input_data)

            logistic_prediction = logistic_model.predict(input_data_scaled)
            logistic_prediction_proba = logistic_model.predict_proba(input_data_scaled)[0][1] * 100

            random_forest_prediction = random_forest_model.predict(input_data_scaled)
            random_forest_prediction_proba = random_forest_model.predict_proba(input_data_scaled)[0][1] * 100

            decision_tree_prediction = decision_tree_model.predict(input_data_scaled)
            decision_tree_prediction_proba = decision_tree_model.predict_proba(input_data_scaled)[0][1] * 100

            svm_prediction = svm_model.predict(input_data_scaled)
            svm_prediction_proba = svm_model.predict_proba(input_data_scaled)[0][1] * 100

            knn_prediction = knn_model.predict(input_data_scaled)
            knn_prediction_proba = knn_model.predict_proba(input_data_scaled)[0][1] * 100

            gradient_boosting_prediction = gradient_boosting_model.predict(input_data_scaled)
            gradient_boosting_prediction_proba = gradient_boosting_model.predict_proba(input_data_scaled)[0][1] * 100

            result = {
                'logistic': f'Logistic Regression: {"DIABETIC" if logistic_prediction == 1 else "NOT DIABETIC"}\n'
                            f'Confidence: {logistic_prediction_proba:.2f}%',
                'random_forest': f'Random Forest: {"DIABETIC" if random_forest_prediction == 1 else "NOT DIABETIC"}\n'
                                 f'Confidence: {random_forest_prediction_proba:.2f}%',
                'decision_tree': f'Decision Tree: {"DIABETIC" if decision_tree_prediction == 1 else "NOT DIABETIC"}\n'
                                 f'Confidence: {decision_tree_prediction_proba:.2f}%',
                'svm': f'SVM: {"DIABETIC" if svm_prediction == 1 else "NOT DIABETIC"}\n'
                       f'Confidence: {svm_prediction_proba:.2f}%',
                'knn': f'KNN: {"DIABETIC" if knn_prediction == 1 else "NOT DIABETIC"}\n'
                       f'Confidence: {knn_prediction_proba:.2f}%',
                'gradient_boosting': f'Gradient Boosting: {"DIABETIC" if gradient_boosting_prediction == 1 else "NOT DIABETIC"}\n'
                                     f'Confidence: {gradient_boosting_prediction_proba:.2f}%'
            }

            session['result'] = result
            return redirect(url_for('result'))
        except ValueError:
            flash("Please enter valid values", 'danger')
    return render_template('predict.html')


@app.route('/result')
@login_required
def result():
    result = session.get('result', None)
    if result is None:
        return redirect(url_for('predict'))
    return render_template('result.html', result=result, logistic_accuracy=logistic_accuracy,
                           random_forest_accuracy=random_forest_accuracy, decision_tree_accuracy=decision_tree_accuracy,
                           svm_accuracy=svm_accuracy, knn_accuracy=knn_accuracy,
                           gradient_boosting_accuracy=gradient_boosting_accuracy, accuracy_graph=accuracy_graph)


@app.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    flash(f'You have successfully logged out, {username}', 'success')
    return redirect(url_for('layout'))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
