import numpy as np
import pandas as pd
import joblib

# Import data
df_data = pd.read_csv('data/data.csv', sep=',', dtype=np.float64)
features_all = df_data.drop('COFs_name',axis=1)

# Import Model 
model = joblib.load('predict/model.pkl')
predict_target = []

# Prediction of results
results = model.predict(features_all)
for i in results:
    predict_target.append(i)
df_data["predict_target"] = predict_target

# Data export
df_data.to_csv("predict/data_predict_target.csv",encoding='utf-8')

