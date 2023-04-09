import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline

def deploy_model(data, model_path):
    # Load pre-trained model
    model = load(model_path)

    # Use pre-processing pipeline to transform data
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    transformed_data = preprocessor.fit_transform(data)

    # Make predictions using the pre-trained model
    predictions = model.predict(transformed_data)

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({'patient_id': data['patient_id'], 'prediction': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    print('Model successfully deployed and predictions saved to predictions.csv.')
