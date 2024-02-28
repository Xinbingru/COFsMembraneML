import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -2.027680835203765
# r2_testing = 0.8276829902346657
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=make_pipeline(
            ZeroCount(),
            RidgeCV()
        )),
        PCA(iterated_power=4, svd_solver="randomized")
    ),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="huber", max_depth=5, max_features=1.0, min_samples_leaf=2, min_samples_split=18, n_estimators=100, subsample=0.7000000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
