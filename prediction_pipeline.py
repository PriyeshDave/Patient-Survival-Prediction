import pandas as pd
import joblib


def predict(data1, data2):
  # Processing test dataset
  for col in data2.columns:
    le = joblib.load('../LabelEncoders/{}_le.pkl'.format(col))
    data2[col] = le.transform(data2[col])
  
  final_df = pd.concat([data1.loc[:,1:], data2], axis=1)

  # Loading trained model
  model = joblib.load('../Model/model_1.pkl')

  # Predicting
  y_pred = model.predict(final_df)
  y_pred = (y_pred > 0.5)

  mapper = {0:'Will Survived', 1:'Will not survive'}
  y_pred = y_pred.map(mapper)

  results = pd.concat([data1.loc[:,1], y_pred], axis=1)
  return results

