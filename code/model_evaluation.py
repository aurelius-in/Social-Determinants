import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Passes in the trained model and testing set X_test and y_test. We use the predict method of the model 
# to generate predictions on the testing set, and then calculate several evaluation metrics, including 
# accuracy, precision, recall, f1 score, and the confusion matrix. These evaluation metrics are returned 
# as a dictionary for further analysis. You can modify this implementation based on the specific requirements 
# of your project, such as using different evaluation metrics or modifying the output format.

def evaluate_model(model, X_test, y_test):
    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Create a dictionary of the evaluation metrics
    evaluation_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': confusion}

    return evaluation_metrics
