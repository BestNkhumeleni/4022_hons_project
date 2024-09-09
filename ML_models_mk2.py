import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
smote = SMOTE()
# Load the data for resolution prediction
X_resolution = pd.read_csv('features_resolution.csv')  # Input features for resolution
y_resolution = pd.read_csv('labels_resolution.csv')    # Labels for resolution
# Load the data for fps prediction
X_fps = pd.read_csv('features_fps.csv')  # Input features for fps
y_fps = pd.read_csv('labels_fps.csv')    # Labels for fps

y_fps = y_fps.drop(columns=['emp'])
X_resolution = X_resolution.drop(columns=['index'])
y_resolution = y_resolution.drop(columns=['index'])
y_resolution = y_resolution.drop(columns=['emp'])
X_fps = X_fps.drop(columns=['index'])
y_fps = y_fps.drop(columns=['index'])

X_resolution, y_resolution = smote.fit_resample(X_resolution, y_resolution) #generates synthetic data using smote.
X_fps, y_fps = smote.fit_resample(X_fps, y_fps) #generates synthetic data using smote.
# scaler = StandardScaler()
# X_resolution_scaled = scaler.fit_transform(X_resolution)
# X_fps_scaled = scaler.fit_transform(X_fps)

# Train-test split for resolution
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resolution, y_resolution, test_size=0.3, random_state=42)

# Train-test split for fps
X_train_fps, X_test_fps, y_train_fps, y_test_fps = train_test_split(X_fps, y_fps, test_size=0.3, random_state=42)

# Define the models
models_resolution = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=100000)
}

models_fps = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=5000)
}

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, task_name, models):
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())  # Flatten y for training
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        print(f"{task_name} - {name} Accuracy: {accuracy:.4f}")
    return trained_models

# Train and evaluate models for resolution prediction
print("Resolution Prediction:")
trained_resolution_models = train_and_evaluate(X_train_res, X_test_res, y_train_res, y_test_res, "Resolution", models_resolution)

# Train and evaluate models for fps prediction
print("\nFPS Prediction:")
trained_fps_models = train_and_evaluate(X_train_fps, X_test_fps, y_train_fps, y_test_fps, "FPS", models_fps)

# Function to predict resolution and fps for a new input
def predict_resolution_and_fps(input_csv, resolution_models, fps_models):
    # Load the single-row input CSV
    input_data = pd.read_csv(input_csv)
    
    # Drop the video name column for prediction
    input_features = input_data.drop(columns=['name'])
    
    # Predict resolution using all trained models
    print("\nResolution Predictions:")
    for name, model in resolution_models.items():
        resolution_prediction = model.predict(input_features)[0]
        print(f"{name} predicts Resolution: {resolution_prediction}")
    
    # Predict fps using all trained models
    print("\nFPS Predictions:")
    for name, model in fps_models.items():
        fps_prediction = model.predict(input_features)[0]
        print(f"{name} predicts FPS: {fps_prediction}")

# Example usage with your new input CSV
# Let's assume you want to use the Random Forest models for prediction.
#print(trained_resolution_models)
predict_resolution_and_fps('/home/best/Desktop/EEE4022S/Data/testing_data/test_720p_30.csv', trained_resolution_models, trained_fps_models)
