import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def predict_from_csv(input_csv):
    # Initialize SMOTE
    smote = SMOTE()

    # Load the training data for resolution and fps prediction
    X_resolution = pd.read_csv('features_resolution.csv')  # Input features for resolution
    y_resolution = pd.read_csv('labels_resolution.csv')    # Labels for resolution
    X_fps = pd.read_csv('features_fps.csv')  # Input features for fps
    y_fps = pd.read_csv('labels_fps.csv')    # Labels for fps

    # Drop unnecessary columns from the datasets
    y_fps = y_fps.drop(columns=['emp'])
    X_resolution = X_resolution.drop(columns=['index'])
    y_resolution = y_resolution.drop(columns=['index', 'emp'])
    X_fps = X_fps.drop(columns=['index'])
    y_fps = y_fps.drop(columns=['index'])

    # Apply SMOTE for resolution data
    X_resolution, y_resolution = smote.fit_resample(X_resolution, y_resolution)
    # X_fps, y_fps = smote.fit_resample(X_fps, y_fps)  # Uncomment for fps if needed

    # Apply StandardScaler for feature scaling
    scaler = StandardScaler()
    X_resolution_scaled = scaler.fit_transform(X_resolution)
    X_fps_scaled = scaler.fit_transform(X_fps)

    # Train-test split for both resolution and fps prediction
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resolution_scaled, y_resolution, test_size=0.3, random_state=42)
    X_train_fps, X_test_fps, y_train_fps, y_test_fps = train_test_split(X_fps_scaled, y_fps, test_size=0.3, random_state=42)

    # Define models for both resolution and fps prediction
    models_resolution = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=5000)
    }
    models_fps = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=5000)
    }

    # Train and evaluate models function
    def train_and_evaluate(X_train, X_test, y_train, y_test, models):
        trained_models = {}
        model_accuracies = {}
        for name, model in models.items():
            model.fit(X_train, y_train.values.ravel())  # Flatten y for training
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            trained_models[name] = model
            model_accuracies[name] = accuracy
            print(f"{name} Accuracy: {accuracy:.4f}")
        # Find the model with the highest accuracy
        best_model_name = max(model_accuracies, key=model_accuracies.get)
        best_model = trained_models[best_model_name]
        return best_model, trained_models

    # Train models for resolution prediction and get the best one
    print("Training Resolution Models:")
    best_resolution_model, trained_resolution_models = train_and_evaluate(X_train_res, X_test_res, y_train_res, y_test_res, models_resolution)

    # Train models for fps prediction and get the best one
    print("\nTraining FPS Models:")
    best_fps_model, trained_fps_models = train_and_evaluate(X_train_fps, X_test_fps, y_train_fps, y_test_fps, models_fps)

    # Load the new input data for prediction
    input_data = pd.read_csv(input_csv)

    # Drop the video name (or any unnecessary) column for prediction
    input_features = input_data.drop(columns=['name'])

    # Scale the input data for prediction
    input_features_scaled = scaler.transform(input_features)

    # Predict resolution using the best model
    resolution_prediction = best_resolution_model.predict(input_features_scaled)[0]
    print(f"\nBest Model predicts Resolution: {resolution_prediction}")

    # Predict fps using the best model
    fps_prediction = best_fps_model.predict(input_features_scaled)[0]
    print(f"Best Model predicts FPS: {fps_prediction}")

    # Return the best predictions
    return resolution_prediction, fps_prediction

# Example usage
predicted_resolution, predicted_fps = predict_from_csv('/home/best/Desktop/EEE4022S/Data/testing_data/test_2_1080p_60.csv')
print(f"\nFinal Predicted Resolution: {predicted_resolution}")
print(f"Final Predicted FPS: {predicted_fps}")
