import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from tpot.export_utils import set_param_recursive
from tpot import TPOTRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt


# Import data
df_data = pd.read_csv('data/data.csv', sep=',', dtype=np.float64)
tpot_data = df_data.drop('COFs_name',axis=1)
features_all = tpot_data.drop(labels=['LCD(Å)','TE'],axis=1)

# Data segmentation
training_features, testing_features, training_target, testing_target = \
            train_test_split(features_all, tpot_data['Target'],test_size=0.2, random_state=42)

# Run tpot
generations=100  #(0,100]
population_size=100  #(0,100]
cv=5
random_state=42
# If you think the runtime is too long, change the parameter to: generations=5, population_size=100

# GENERATIONS and POPULATION_SIZE determine tpot runtime and prediction accuracy, see the paper for details: Olson R S, Moore J H. Automated Machine Learning, 2019: 151.
tpot = TPOTRegressor(generations=generations, population_size=population_size, 
                     verbosity=2,scoring='neg_mean_squared_error',
                     cv=cv,random_state=random_state)

# Calculate R2
tpot.fit(training_features, training_target)
preds = tpot.predict(testing_features)
print("r2_testing =",r2_score(testing_target, preds))

# Print Code
tpot.export('model/model.py')
print(tpot.export())