import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Function to load and prepare the dataset
def load_and_prepare_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Handle missing values, if any (filling with median or other methods)
    data = data.fillna(data.median())  # More sophisticated filling strategy
    
    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Adjust the target variable (make sure it's binary)
    y = (y == 1).astype(int)
    
    return X, y

# Function to split the dataset into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Function to perform feature scaling
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to train and evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    
    return accuracy, auc, conf_matrix, class_report

# Function to perform hyperparameter tuning (optional for improvement)
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Function to visualize the model comparison
def plot_model_comparison(models, accuracy_scores, auc_scores):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the accuracy scores as a bar chart
    ax.bar(models, accuracy_scores, color='skyblue', label='Accuracy', alpha=0.7)
    ax.set_ylabel('Accuracy', color='blue')
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison')

    # Plotting the AUC scores on a secondary axis
    ax2 = ax.twinx()
    ax2.plot(models, auc_scores, color='orange', marker='o', label='AUC Score', linewidth=2)
    ax2.set_ylabel('AUC Score', color='orange')
    ax2.set_ylim(0, 1)

    # Adding legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot
    plt.show()

# Function to visualize confusion matrix
def plot_confusion_matrix(conf_matrix, model_name):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Main function to run the project
def main():
    # Load and prepare the dataset
    file_path = 'C:/Users/shravani/mypython/synthetic_sensor_data.csv'
    X, y = load_and_prepare_data(file_path)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Feature scaling
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # List to store model performances
    models = ['Random Forest', 'SVM', 'Naïve Bayes', 'XGBoost']
    accuracy_scores = []
    auc_scores = []
    
    # Hyperparameters grid (for tuning)
    rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    xgb_param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    
    # Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    rf_best = tune_hyperparameters(rf, rf_param_grid, X_train_scaled, y_train)  # Hyperparameter tuning
    rf_accuracy, rf_auc, rf_conf_matrix, rf_class_report = evaluate_model(rf_best, X_train_scaled, X_test_scaled, y_train, y_test)
    accuracy_scores.append(rf_accuracy)
    auc_scores.append(rf_auc)
    print(f'Random Forest Accuracy: {rf_accuracy:.4f}, AUC Score: {rf_auc:.4f}')
    print(f'Random Forest Classification Report:\n{rf_class_report}')
    plot_confusion_matrix(rf_conf_matrix, 'Random Forest')

    # Support Vector Machine (SVM)
    svm = SVC(probability=True, random_state=42)
    svm_best = tune_hyperparameters(svm, svm_param_grid, X_train_scaled, y_train)  # Hyperparameter tuning
    svm_accuracy, svm_auc, svm_conf_matrix, svm_class_report = evaluate_model(svm_best, X_train_scaled, X_test_scaled, y_train, y_test)
    accuracy_scores.append(svm_accuracy)
    auc_scores.append(svm_auc)
    print(f'SVM Accuracy: {svm_accuracy:.4f}, AUC Score: {svm_auc:.4f}')
    print(f'SVM Classification Report:\n{svm_class_report}')
    plot_confusion_matrix(svm_conf_matrix, 'SVM')

    # Naïve Bayes Classifier
    nb = GaussianNB()
    nb_accuracy, nb_auc, nb_conf_matrix, nb_class_report = evaluate_model(nb, X_train_scaled, X_test_scaled, y_train, y_test)
    accuracy_scores.append(nb_accuracy)
    auc_scores.append(nb_auc)
    print(f'Naïve Bayes Accuracy: {nb_accuracy:.4f}, AUC Score: {nb_auc:.4f}')
    print(f'Naïve Bayes Classification Report:\n{nb_class_report}')
    plot_confusion_matrix(nb_conf_matrix, 'Naïve Bayes')

    # XGBoost Classifier (without hyperparameter tuning to bypass the error)
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_accuracy, xgb_auc, xgb_conf_matrix, xgb_class_report = evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test)
    accuracy_scores.append(xgb_accuracy)
    auc_scores.append(xgb_auc)
    print(f'XGBoost Accuracy: {xgb_accuracy:.4f}, AUC Score: {xgb_auc:.4f}')
    print(f'XGBoost Classification Report:\n{xgb_class_report}')
    plot_confusion_matrix(xgb_conf_matrix, 'XGBoost')

    # Visualize the comparison of models
    plot_model_comparison(models, accuracy_scores, auc_scores)

if __name__ == '__main__':
    main()
