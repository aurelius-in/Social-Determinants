import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# The select_model() function performs a grid search over multiple machine 
# learning models to find the best model for the data. In this example, 
# we're using logistic regression, decision trees, random forests, and 
# neural networks. For each model, we define a set of hyperparameters to 
# search over using GridSearchCV or RandomizedSearchCV. We then fit each 
# model with the training data and record the best score and best hyperparameters. 
# Finally, we return a dataframe of the results sorted by best score.

def select_model(X_train, y_train):
    '''
    This function performs a grid search over multiple machine learning models to find the best model for the data.
    '''
    models = {
        'logistic_regression': {
            'model': LogisticRegression(),
            'params': {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 1, 10]
            }
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [5, 10, 15],
                'min_samples_leaf': [1, 2, 5, 10]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50, 100],
                'max_depth': [5, 10, 15],
                'min_samples_leaf': [1, 2, 5, 10]
            }
        },
        'neural_network': {
            'model': MLPClassifier(),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    }
    
    scores = []
    for model_name, model_params in models.items():
        clf = RandomizedSearchCV(model_params['model'], model_params['params'], cv=5, n_iter=10)
        clf.fit(X_train, y_train)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
        
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params']).sort_values('best_score', ascending=False)
