import pandas as pd
import numpy as np

# The load_data() function reads in the data from a CSV file and returns a pandas dataframe.
# The clean_data() function cleans the data by dropping irrelevant columns, filling missing values,
# and converting categorical variables to numerical values.

# The preprocess_data() function applies standard scaling to the data and splits it into training 
# and testing sets, which will be used for model training and evaluation.

# Note that this is just an example, and the actual preprocessing steps may differ depending on the data and the specific problem at hand.

def load_data(file_path):
    '''
    This function loads the data from a CSV file and returns a pandas dataframe.
    '''
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    '''
    This function cleans the data by dropping irrelevant columns, filling missing values, and converting categorical variables to numerical values.
    '''
    # Drop irrelevant columns
    data = data.drop(['id', 'date'], axis=1)
    
    # Fill missing values with median
    data = data.fillna(data.median())
    
    # Convert categorical variables to numerical values
    data['sex'] = np.where(data['sex'] == 'F', 1, 0)
    data['smoker'] = np.where(data['smoker'] == 'yes', 1, 0)
    data['region'] = data['region'].map({'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3})
    
    return data

def preprocess_data(data):
    '''
    This function applies standard scaling to the data and splits it into training and testing sets.
    '''
    # Apply standard scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, data['charges'], test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
