# MLFlow Example Notebook

## Repo: https://github.com/SANDataHPE/MLFlow-Examples/tree/main/Bike-Sharing-MLFlow-Example
---

This notebook demonstrates an example of dataset preprocessing, ML model training and evaluation, model tuning via MLflow tracking and finally REST API model serving via MLflow models.

---
- **Dateset:** Bike Sharing Dataset: http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip
- **Goal:** predict `rented_bikes` (count per hour) based on weather and time information.


**References:**
- https://docs.databricks.com/_static/notebooks/gbt-regression.html
- https://www.kaggle.com/pratsiuk/mlflow-experiment-automation-top-9
- https://mlflow.org/docs/latest/tracking.html

# Setup ML Flow Magics

### Load ML Flow Enviornment variable


```python
%loadMlflow
```

    Backend configured


### Set ML Flow Experiment Name


```python
%Setexp --name bikesharetesting
```

    INFO: 'bikesharetesting' does not exist. Creating a new experiment


# Install Required Libraries and Python Packages


```python
#!export http_proxy=http://proxy:8080 
#!export https_proxy=http:/proxy:8080 
#!conda install -c anaconda graphviz
#!pip install --proxy http://proxy:8080 kfp==1.4.0 pydotplus
```

# Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifact

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from mlflow.models.signature import infer_signature
from sklearn import tree

from pydotplus import graph_from_dot_data
import graphviz
from IPython.display import Image

import itertools

plt.style.use("fivethirtyeight")
pd.plotting.register_matplotlib_converters()

import warnings
warnings.filterwarnings('ignore')
```

# Import Data

Dataset and explanation:
http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

- Input file: `hour.csv` - contains bike sharing counts aggregated on hourly basis. 
- Size: 17379 hours / rows



```python
# download and extract csv files into Data folder
#!wget -nc "http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
#!unzip -o "Bike-Sharing-Dataset.zip"
```


```python
# load input data into pandas dataframe
bike_sharing = pd.read_csv("hour.csv")
bike_sharing        
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instant</th>
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0000</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17374</th>
      <td>17375</td>
      <td>2012-12-31</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.26</td>
      <td>0.2576</td>
      <td>0.60</td>
      <td>0.1642</td>
      <td>11</td>
      <td>108</td>
      <td>119</td>
    </tr>
    <tr>
      <th>17375</th>
      <td>17376</td>
      <td>2012-12-31</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.26</td>
      <td>0.2576</td>
      <td>0.60</td>
      <td>0.1642</td>
      <td>8</td>
      <td>81</td>
      <td>89</td>
    </tr>
    <tr>
      <th>17376</th>
      <td>17377</td>
      <td>2012-12-31</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>21</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.26</td>
      <td>0.2576</td>
      <td>0.60</td>
      <td>0.1642</td>
      <td>7</td>
      <td>83</td>
      <td>90</td>
    </tr>
    <tr>
      <th>17377</th>
      <td>17378</td>
      <td>2012-12-31</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.26</td>
      <td>0.2727</td>
      <td>0.56</td>
      <td>0.1343</td>
      <td>13</td>
      <td>48</td>
      <td>61</td>
    </tr>
    <tr>
      <th>17378</th>
      <td>17379</td>
      <td>2012-12-31</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.26</td>
      <td>0.2727</td>
      <td>0.65</td>
      <td>0.1343</td>
      <td>12</td>
      <td>37</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
<p>17379 rows × 17 columns</p>
</div>



## Data preprocessing


```python
# remove unused columns
bike_sharing.drop(columns=["instant", "dteday", "registered", "casual"], inplace=True)

# use better names
bike_sharing.rename(
    columns={
        "yr": "year",
        "mnth": "month",
        "hr": "hour_of_day",
        "holiday": "is_holiday",
        "workingday": "is_workingday",
        "weathersit": "weather_situation",
        "temp": "temperature",
        "atemp": "feels_like_temperature",
        "hum": "humidity",
        "cnt": "rented_bikes",
    },
    inplace=True,
)

# show samples
bike_sharing
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>year</th>
      <th>month</th>
      <th>hour_of_day</th>
      <th>is_holiday</th>
      <th>weekday</th>
      <th>is_workingday</th>
      <th>weather_situation</th>
      <th>temperature</th>
      <th>feels_like_temperature</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>rented_bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0000</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17374</th>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.26</td>
      <td>0.2576</td>
      <td>0.60</td>
      <td>0.1642</td>
      <td>119</td>
    </tr>
    <tr>
      <th>17375</th>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.26</td>
      <td>0.2576</td>
      <td>0.60</td>
      <td>0.1642</td>
      <td>89</td>
    </tr>
    <tr>
      <th>17376</th>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>21</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.26</td>
      <td>0.2576</td>
      <td>0.60</td>
      <td>0.1642</td>
      <td>90</td>
    </tr>
    <tr>
      <th>17377</th>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.26</td>
      <td>0.2727</td>
      <td>0.56</td>
      <td>0.1343</td>
      <td>61</td>
    </tr>
    <tr>
      <th>17378</th>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.26</td>
      <td>0.2727</td>
      <td>0.65</td>
      <td>0.1343</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
<p>17379 rows × 13 columns</p>
</div>



### Data Visualization 


```python
hour_of_day_agg = bike_sharing.groupby(["hour_of_day"])["rented_bikes"].sum()

hour_of_day_agg.plot(
    kind="line", 
    title="Total rented bikes by hour of day",
    xticks=hour_of_day_agg.index,
    figsize=(15, 10),
)
```




    <AxesSubplot:title={'center':'Total rented bikes by hour of day'}, xlabel='hour_of_day'>




    
![png](output_16_1.png)
    


## Prepare training and test data sets



```python
# Split the dataset randomly into 70% for training and 30% for testing.
X = bike_sharing.drop("rented_bikes", axis=1)
y = bike_sharing.rented_bikes
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

print(f"Training samples: {X_train.size}")
print(f"Test samples: {X_test.size}")
```

    Training samples: 145980
    Test samples: 62568


# Evaluation Metrics

Create evaluation methods to be used in training stage (next step)

## Root Mean Square Error (RMSE)

References: 
- https://medium.com/@xaviergeerinck/artificial-intelligence-how-to-measure-performance-accuracy-precision-recall-f1-roc-rmse-611d10e4caac
- https://www.kaggle.com/residentmario/model-fit-metrics#Root-mean-squared-error-(RMSE)



```python
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def rmse_score(y, y_pred):
    score = rmse(y, y_pred)
    print("RMSE score: {:.4f}".format(score))
    return score
```

### Cross-Validation RMSLE score

cross-validation combines (averages) measures of fitness in prediction to derive a more accurate estimate of model prediction performance.

Background: 
- https://en.wikipedia.org/wiki/Cross-validation_(statistics)
- https://www.kaggle.com/carlolepelaars/understanding-the-metric-rmsle



```python
def rmsle_cv(model, X_train, y_train):
    kf = KFold(n_splits=3, shuffle=True, random_state=42).get_n_splits(X_train.values)
    # Evaluate a score by cross-validation
    rmse = np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse


def rmse_cv_score(model, X_train, y_train):
    score = rmsle_cv(model, X_train, y_train)
    print("Cross-Validation RMSE score: {:.4f} (std = {:.4f})".format(score.mean(), score.std()))
    return score
```

## Feature Importance

Background: https://medium.com/bigdatarepublic/feature-importance-whats-in-a-name-79532e59eea3


```python
def model_feature_importance(model):
    feature_importance = pd.DataFrame(
        model.feature_importances_,
        index=X_train.columns,
        columns=["Importance"],
    )

    # sort by importance
    feature_importance.sort_values(by="Importance", ascending=False, inplace=True)

    # plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=feature_importance.reset_index(),
        y="index",
        x="Importance",
    ).set_title("Feature Importance")
    # save image
    plt.savefig("model_artifacts/feature_importance.png", bbox_inches='tight')
```

## Permutation Importance

Background: https://www.kaggle.com/dansbecker/permutation-importance


```python
def model_permutation_importance(model):
    p_importance = permutation_importance(model, X_test, y_test, random_state=42, n_jobs=-1)

    # sort by importance
    sorted_idx = p_importance.importances_mean.argsort()[::-1]
    p_importance = pd.DataFrame(
        data=p_importance.importances[sorted_idx].T,
        columns=X_train.columns[sorted_idx]
    )

    # plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=p_importance,
        orient="h"
    ).set_title("Permutation Importance")

    # save image
    plt.savefig("model_artifacts/permutation_importance.png", bbox_inches="tight")
```

## Decision Tree Visualization

Reference: https://towardsdatascience.com/visualizing-decision-trees-with-python-scikit-learn-graphviz-matplotlib-1c50b4aa68dc 


TODO: plot all trees


```python
def model_tree_visualization(model):
    # generate visualization
    tree_dot_data = tree.export_graphviz(
        decision_tree=model.estimators_[0, 0],  # Get the first tree,
        label="all",
        feature_names=X_train.columns,
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        precision=1,
    )

    # save image
    graph_from_dot_data(tree_dot_data).write_png("model_artifacts/Decision_Tree_Visualization.png")

    # show tree
    return graphviz.Source(tree_dot_data)
```

# MLflow Tracking

Reference: https://www.mlflow.org/docs/latest/cli.html#mlflow-ui


## MLflow Logger


```python
# Track params and metrics
def log_mlflow_run(model, signature):
    # Auto-logging for scikit-learn estimators
    # mlflow.sklearn.autolog()

    # log estimator_name name
    name = model.__class__.__name__
    mlflow.set_tag("estimator_name", name)

    # log input features
    mlflow.set_tag("features", str(X_train.columns.values.tolist()))

    # Log tracked parameters only
    mlflow.log_params({key: model.get_params()[key] for key in parameters})

    mlflow.log_metrics({
        'RMSE_CV': score_cv.mean(),
        'RMSE': score,
    })

    # log training loss
    for s in model.train_score_:
        mlflow.log_metric("Train Loss", s)

    # Save model to artifacts
    mlflow.sklearn.log_model(model, "model", signature=signature)

    # log charts
    mlflow.log_artifacts("model_artifacts")

    # misc
    # Log all model parameters
    mlflow.log_params(model.get_params())
    mlflow.log_param("Training size", X_test.size)
    mlflow.log_param("Test size", y_test.size)
```

# Model Training


## Model Type & Method

For this example,
- Approache: Decision tree (Supervised learning)
- Tree type: Regression tree
- Technique/ensemble method: Gradient boosting

**All put together we get:** [GBRT (Gradient Boosted Regression Tree)](https://orbi.uliege.be/bitstream/2268/163521/1/slides.pdf)

Background:
- Choosing a model: https://scikit-learn.org/stable/tutorial/machine_learning_map
- Machine Learning Models Explained
: https://docs.paperspace.com/machine-learning/wiki/machine-learning-models-explained
- Gradient Boosted Regression Trees: https://orbi.uliege.be/bitstream/2268/163521/1/slides.pdf



```python
# GBRT (Gradient Boosted Regression Tree) scikit-learn implementation 
model_class = GradientBoostingRegressor
```

## Model Hyper-parameters 



```python
parameters = {
    "learning_rate": [0.1, 0.05, 0.01],
    "max_depth": [4, 5, 6],
    # "verbose": True,
}
```

### Tuning the hyper-parameters: Grid search

- Simple but inefficient
- more advanced tuning techniques: https://research.fb.com/efficient-tuning-of-online-systems-using-bayesian-optimization/


```python
# generate parameters combinations
params_keys = parameters.keys()
params_values = [
    parameters[key] if isinstance(parameters[key], list) else [parameters[key]]
    for key in params_keys
]
runs_parameters = [
    dict(zip(params_keys, combination)) for combination in itertools.product(*params_values)
]
```

## Training runs


```python
# training loop
for i, run_parameters in enumerate(runs_parameters):
    print(f"Run {i}: {run_parameters}")

    # mlflow: stop active runs if any
    if mlflow.active_run():
        mlflow.end_run()
    # mlflow:track run
    mlflow.start_run(run_name=f"Run {i}")

    # create model instance
    model = model_class(**run_parameters)

    # train
    model.fit(X_train, y_train)

    # get evaluations scores
    score = rmse_score(y_test, model.predict(X_test))
    score_cv = rmse_cv_score(model, X_train, y_train)
    
    # generate charts
    model_feature_importance(model)
    plt.close()
    model_permutation_importance(model)
    plt.close()
    model_tree_visualization(model)

    # get model signature
    signature = infer_signature(model_input=X_train, model_output=model.predict(X_train))

    # mlflow: log metrics
    log_mlflow_run(model, signature)

    # mlflow: end tracking
    mlflow.end_run()
    print("")
```

    Run 0: {'learning_rate': 0.1, 'max_depth': 4}
    RMSE score: 52.0019
    Cross-Validation RMSE score: 56.5502 (std = 0.1427)
    
    Run 1: {'learning_rate': 0.1, 'max_depth': 5}
    RMSE score: 44.6961
    Cross-Validation RMSE score: 48.1914 (std = 0.1712)
    
    Run 2: {'learning_rate': 0.1, 'max_depth': 6}
    RMSE score: 41.8623
    Cross-Validation RMSE score: 44.9797 (std = 0.4243)
    
    Run 3: {'learning_rate': 0.05, 'max_depth': 4}
    RMSE score: 63.1557
    Cross-Validation RMSE score: 67.8193 (std = 1.7241)
    
    Run 4: {'learning_rate': 0.05, 'max_depth': 5}
    RMSE score: 53.0504
    Cross-Validation RMSE score: 55.9904 (std = 0.8639)
    
    Run 5: {'learning_rate': 0.05, 'max_depth': 6}
    RMSE score: 46.2968
    Cross-Validation RMSE score: 49.8090 (std = 0.5420)
    
    Run 6: {'learning_rate': 0.01, 'max_depth': 4}
    RMSE score: 120.1656
    Cross-Validation RMSE score: 123.9446 (std = 1.0540)
    
    Run 7: {'learning_rate': 0.01, 'max_depth': 5}
    RMSE score: 112.4922
    Cross-Validation RMSE score: 116.0879 (std = 0.9958)
    
    Run 8: {'learning_rate': 0.01, 'max_depth': 6}
    RMSE score: 106.2637
    Cross-Validation RMSE score: 109.3513 (std = 1.2057)
    


# Best Model Results


```python
best_run_df = mlflow.search_runs(order_by=['metrics.RMSE_CV ASC'], max_results=1)
if len(best_run_df.index) == 0:
    raise Exception(f"Found no runs for experiment '{experiment_name}'")

best_run = mlflow.get_run(best_run_df.at[0, 'run_id'])
best_model_uri = f"{best_run.info.artifact_uri}/model"
best_model = mlflow.sklearn.load_model(best_model_uri)
```


```python
# print best run info
print("Best run info:")
print(f"Run id: {best_run.info.run_id}")
print(f"Run parameters: {best_run.data.params}")
print("Run score: RMSE_CV = {:.4f}".format(best_run.data.metrics['RMSE_CV']))
print(f"Run model URI: {best_model_uri}")
```

    Best run info:
    Run id: d52a00c543e94d26a37994a517fce096
    Run parameters: {'min_impurity_decrease': '0.0', 'verbose': '0', 'random_state': 'None', 'min_samples_leaf': '1', 'max_features': 'None', 'n_iter_no_change': 'None', 'Training size': '62568', 'ccp_alpha': '0.0', 'Test size': '5214', 'init': 'None', 'min_weight_fraction_leaf': '0.0', 'criterion': 'friedman_mse', 'tol': '0.0001', 'validation_fraction': '0.1', 'min_samples_split': '2', 'max_depth': '6', 'warm_start': 'False', 'max_leaf_nodes': 'None', 'learning_rate': '0.1', 'n_estimators': '100', 'alpha': '0.9', 'presort': 'deprecated', 'loss': 'ls', 'min_impurity_split': 'None', 'subsample': '1.0'}
    Run score: RMSE_CV = 44.9797
    Run model URI: s3://mlflow/10/d52a00c543e94d26a37994a517fce096/artifacts/model



```python
model_feature_importance(best_model)
```


    
![png](output_45_0.png)
    



```python
model_permutation_importance(best_model)
```


    
![png](output_46_0.png)
    



```python
model_tree_visualization(best_model)
```




    
![svg](output_47_0.svg)
    



# Inference


```python
test_predictions = X_test.copy()
# real output (rented_bikes) from test dataset
test_predictions["rented_bikes"] = y_test

# add "predicted_rented_bikes" from test dataset
test_predictions["predicted_rented_bikes"] = best_model.predict(X_test).astype(int)

# show results
test_predictions
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>year</th>
      <th>month</th>
      <th>hour_of_day</th>
      <th>is_holiday</th>
      <th>weekday</th>
      <th>is_workingday</th>
      <th>weather_situation</th>
      <th>temperature</th>
      <th>feels_like_temperature</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>rented_bikes</th>
      <th>predicted_rented_bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12830</th>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>19</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.80</td>
      <td>0.6970</td>
      <td>0.27</td>
      <td>0.1940</td>
      <td>425</td>
      <td>397</td>
    </tr>
    <tr>
      <th>8688</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>20</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2273</td>
      <td>0.41</td>
      <td>0.2239</td>
      <td>88</td>
      <td>99</td>
    </tr>
    <tr>
      <th>7091</th>
      <td>4</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0.32</td>
      <td>0.3030</td>
      <td>0.66</td>
      <td>0.2836</td>
      <td>4</td>
      <td>13</td>
    </tr>
    <tr>
      <th>12230</th>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>19</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.78</td>
      <td>0.7121</td>
      <td>0.52</td>
      <td>0.3582</td>
      <td>526</td>
      <td>564</td>
    </tr>
    <tr>
      <th>431</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0.26</td>
      <td>0.2273</td>
      <td>0.56</td>
      <td>0.3881</td>
      <td>13</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12749</th>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.82</td>
      <td>0.7727</td>
      <td>0.52</td>
      <td>0.1343</td>
      <td>167</td>
      <td>182</td>
    </tr>
    <tr>
      <th>11476</th>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.38</td>
      <td>0.3939</td>
      <td>0.37</td>
      <td>0.0000</td>
      <td>214</td>
      <td>241</td>
    </tr>
    <tr>
      <th>12847</th>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.80</td>
      <td>0.6970</td>
      <td>0.33</td>
      <td>0.2239</td>
      <td>556</td>
      <td>555</td>
    </tr>
    <tr>
      <th>16721</th>
      <td>4</td>
      <td>1</td>
      <td>12</td>
      <td>12</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.52</td>
      <td>0.5000</td>
      <td>0.68</td>
      <td>0.1940</td>
      <td>312</td>
      <td>297</td>
    </tr>
    <tr>
      <th>9511</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.16</td>
      <td>0.1818</td>
      <td>0.86</td>
      <td>0.1045</td>
      <td>72</td>
      <td>74</td>
    </tr>
  </tbody>
</table>
<p>5214 rows × 14 columns</p>
</div>




```python
# plot truth vs prediction values
test_predictions.plot(
    kind="scatter",
    x="rented_bikes",
    y="predicted_rented_bikes",
    title="Rented bikes vs predicted rented bikes",
    figsize=(15, 15),
)
```




    <AxesSubplot:title={'center':'Rented bikes vs predicted rented bikes'}, xlabel='rented_bikes', ylabel='predicted_rented_bikes'>




    
![png](output_50_1.png)
    


# Model Serving

Reference: https://www.mlflow.org/docs/latest/models.html

## Create Secret For Accessing Saved Model From MLFlow Minio in Namespace Where We Are Going To Deploy Seldon Depeployment


```bash
%%bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: seldon-init-container-secret
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: admin # Minio ACCESS KEY
  AWS_SECRET_ACCESS_KEY: admin123 # Minio Secret Key
  AWS_ENDPOINT_URL: http://minio:10032 # Minio Endpoint
  USE_SSL: "false"
EOF
```

### Create KFP Client Object


```python
import kfp
import json
import kfp.dsl as dsl


import requests, kfp

endpoint = "http://kubeflow:10029/pipeline" # Kubeflow Url
api_username  = "***" # Username
api_password = "***" # Password


def get_user_auth_session_cookie(url, username, password):
    url = url.replace('/pipeline', '')
    get_response = requests.get(url)
    if 'auth' in get_response.url:
        credentials = {'login': username, 'password': password}
        # Authenticate user
        session = requests.Session()
        session.post(get_response.url, data=credentials)
        cookie_auth_key = 'authservice_session'
        cookie_auth_value = session.cookies.get(cookie_auth_key)
        if cookie_auth_value:
            return cookie_auth_key + '=' + cookie_auth_value

session_cookie = get_user_auth_session_cookie(endpoint,api_username,api_password)
client = kfp.Client(host=endpoint,cookies=session_cookie)
```

### K8s Resource JSON For Seldon Deployment and passing best_model_uri 


```python
#Namespace Name 
NAMESPACE="st01"

DEPLOYMENT = {
  "apiVersion": "machinelearning.seldon.io/v1alpha3",
  "kind": "SeldonDeployment",
  "metadata": {
    "name": "bikesharing",
    "namespace": NAMESPACE
  },
  "spec": {
    "name": "bikesharing",
    "predictors": [
      {
        "graph": {
          "children": [],
          "implementation": "MLFLOW_SERVER",
          "modelUri": best_model_uri,
          "name": "bikesharing",
          "envSecretRefName": "seldon-init-container-secret"
        },
        "name": "bikesharing",
        "replicas": 1,
        "svcOrchSpec": {
          "resources": {
            "limits": {
              "cpu": "1"
            },
            "requests": {
              "cpu": "0.5"
            }
          }
        },
        "componentSpecs": [
          {
            "spec": {
              "containers": [
                {
                  "resources": {
                    "limits": {
                      "cpu": "1"
                    },
                    "requests": {
                      "cpu": "0.5"
                    }
                  },
                  "env": [
                    {
                      "name": "https_proxy",
                      "value": "proxy:8080"
                    },
                    {
                      "name": "http_proxy",
                      "value": "proxy:8080"
                    }
                  ],
                  "name": "bikesharing",
                  "livenessProbe": {
                    "initialDelaySeconds": 80,
                    "failureThreshold": 200,
                    "periodSeconds": 5,
                    "successThreshold": 1,
                    "httpGet": {
                      "path": "/health/ping",
                      "port": "http",
                      "scheme": "HTTP"
                    }
                  },
                  "readinessProbe": {
                    "initialDelaySeconds": 80,
                    "failureThreshold": 200,
                    "periodSeconds": 5,
                    "successThreshold": 1,
                    "httpGet": {
                      "path": "/health/ping",
                      "port": "http",
                      "scheme": "HTTP"
                    }
                  }
                }
              ]
            }
          }
        ]
      }
    ]
  }
}

DEPLOYMENT = json.dumps(DEPLOYMENT)
```

## Submiting KFP Pipeline Which Will Deploy Trained Model Through Seldon Deployment


```python
@dsl.pipeline(
    name="Deploy Model",
    description="Deploy Trained Model Using Seldon"
)
def deploy_model():
    
    # Deploy mlflow model.
    op_seldondeployment = dsl.ResourceOp(
        name='seldon-deployment',
        k8s_resource=json.loads(DEPLOYMENT),
        action='create'
    )
    
    
client.create_run_from_pipeline_func(
    deploy_model,
    experiment_name="Seldon Deployment",
    arguments={},
)
```


<a href="http://mip-bd-vm182.mip.storage.hpecorp.net:10029/pipeline/#/experiments/details/80da82bd-c8c2-41b6-bfef-87fedeb6783c" target="_blank" >Experiment details</a>.



<a href="http://mip-bd-vm182.mip.storage.hpecorp.net:10029/pipeline/#/runs/details/c1ffea86-2738-423a-a420-1f52de9846bf" target="_blank" >Run details</a>.





    RunPipelineResult(run_id=c1ffea86-2738-423a-a420-1f52de9846bf)



## Prediction


```python
import requests, json

KUBEFLOW_URL = endpoint.replace('/pipeline', '')
SELDON_DEPLOYMENT_NAME="bikesharing"

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

cookies = {
    'authservice_session': session_cookie.replace('authservice_session=', '')
}

json_request= json.dumps({"data":{"ndarray": [[1, 0, 1, 0, 0, 6, 0, 1, 0.24, 0.2879, 0.81, 0.0000]]}})

URL = "{}/seldon/{}/{}/api/v1.0/predictions".format(KUBEFLOW_URL,NAMESPACE,SELDON_DEPLOYMENT_NAME)

response = requests.post(URL, headers=headers, cookies=cookies, data=json_request)
print(response.text)
```

    {"data":{"names":[],"ndarray":[35.616200948055564]},"meta":{}}
    

