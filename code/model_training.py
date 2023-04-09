import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# The train_test_split function from Scikit-learn splits the data into 
# training and testing sets. The logistic regression model is trained on the 
# training set and evaluate its performance on the testing set using the accuracy metric. 
# The train_model function returns the trained model and its accuracy score. 
# Modify for specific requirements, ml algorithms or evaluating the model on different performance metrics.

def train_model(data):
    # Split the data into features and target
    X = data.drop('target_variable', axis=1)
    y = data['target_variable']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy
